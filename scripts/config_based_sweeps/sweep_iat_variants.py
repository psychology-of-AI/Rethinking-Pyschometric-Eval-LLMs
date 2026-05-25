#!/usr/bin/env python3
"""
Run a prompt sweep for the IAT behavior sandbox using a JSON spec (mirrors CCT sweeps).

Spec format (example):
{
  "experiment_name": "iat_prompt_sweep_v1",
  "provider": "openrouter",
  "models_config": "configs/openrouter_models.json",
  "model_keys": ["gpt4o_mini"],
  "grid": {
    "seeds": [42],
    "temperatures": [0.0],
    "top_ps": [1.0],
    "persona_prompts": [""],
    "task_seed": 123
  },
  "max_tokens": 900,
  "store_raw_response": false,
  "iat": {
    "stimuli_json": "configs/iat/iat_stimuli.json",
    "orders_per_test": 3,
    "tests": ["race_racism"]  // optional; default: all tests
  },
  "variants": [
    {
      "variant_id": "neutral_iat",
      "base_system_prompt": "...",
      "task_context": "...",
      "system_prompt": "",           // optional extra system prompt
      "tact": { ... },               // required
      "iat": { ... }                 // optional per-variant IATConfig overrides
    }
  ]
}

Outputs:
  <out_root>/<experiment_name>/<variant_id>/iat_steps.csv
  <out_root>/<experiment_name>/<variant_id>/iat_runs.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from psyai_eval.core.types import GenParams, TACT
from psyai_eval.llms.openai_compat import OpenAICompatChatClient
from psyai_eval.runner.iat_runner import run_iat_once
from psyai_eval.tasks.iat.env import IATConfig, IATEnv


# ---------------------------
# Helpers: config parsing
# ---------------------------

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_slug(s: str, max_len: int = 80) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:max_len] if len(s) > max_len else s


def _persona_label(system_prompt: str) -> str:
    # IMPORTANT: this must match how the runner computes persona_label
    p = (system_prompt or "").strip()
    if p == "":
        return "empty"
    if p == "You are a helpful assistant.":
        return "helpful"
    if p == "Respond to instructions.":
        return "instructions"
    return f"p{abs(hash(p)) % 10_000}"


def _models_from_config(models_cfg: dict, provider: str, model_keys: Optional[List[str]]) -> List[Tuple[str, str]]:
    """
    Returns [(model_key, model_id), ...] for this provider.
    Matches configs/openrouter_models.json format:
      { "openrouter": { "llama31_8b": "...", ... }, "llama": {...} }
    """
    provider = provider.lower().strip()
    if provider not in models_cfg:
        raise ValueError(
            f"Provider '{provider}' not found in models config. Available: {list(models_cfg.keys())}"
        )
    mp = models_cfg[provider]
    if not isinstance(mp, dict):
        raise ValueError(f"models_config['{provider}'] must be an object mapping model_key -> model_id")

    if model_keys:
        pairs = []
        for k in model_keys:
            if k not in mp:
                raise ValueError(f"Model key '{k}' not found under provider '{provider}' in models config.")
            pairs.append((k, str(mp[k])))
        return pairs

    return [(k, str(v)) for k, v in mp.items()]


def _get_openrouter_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    return key


def _get_llama_key() -> str:
    key = os.environ.get("LLAMA_API_KEY", "").strip()
    if not key:
        raise RuntimeError("LLAMA_API_KEY is not set.")
    return key


def _make_client(provider: str, model: str, base_url: Optional[str], headers: dict) -> OpenAICompatChatClient:
    provider = provider.lower().strip()

    if provider == "llama":
        api_key = _get_llama_key()
        base_url = base_url or "https://api.llama-api.com"
        return OpenAICompatChatClient(api_key=api_key, base_url=base_url, model=model)

    if provider == "openrouter":
        api_key = _get_openrouter_key()
        base_url = base_url or "https://openrouter.ai/api/v1"

        default_headers = {}
        if headers.get("HTTP-Referer"):
            default_headers["HTTP-Referer"] = headers["HTTP-Referer"]
        if headers.get("X-Title"):
            default_headers["X-Title"] = headers["X-Title"]

        return OpenAICompatChatClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            default_headers=default_headers or None,
        )

    raise ValueError(f"Unknown provider: {provider}")


# ---------------------------
# Helpers: IAT env config
# ---------------------------

def _filter_dataclass_kwargs(dc_type, d: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {f.name for f in fields(dc_type)}
    return {k: v for k, v in (d or {}).items() if k in allowed}


def _apply_iat_overrides(base_cfg: IATConfig, overrides: Dict[str, Any]) -> IATConfig:
    """
    IATConfig is a frozen dataclass; apply overrides via dataclasses.replace.
    Unknown keys are ignored.
    """
    kw = _filter_dataclass_kwargs(IATConfig, overrides or {})
    return replace(base_cfg, **kw) if kw else base_cfg


def _tact_from_dict(d: Dict[str, Any]) -> TACT:
    if not isinstance(d, dict):
        raise ValueError("variant.tact must be an object")
    return TACT(
        target=str(d.get("target", "I")),
        action=str(d.get("action", "")),
        context=str(d.get("context", "")),
        time=str(d.get("time", "")),
        policy_label=str(d.get("policy_label", "")),
    )


def _select_tests(stimuli: Dict[str, Any], iat_spec: Dict[str, Any]) -> List[str]:
    # Explicit tests list wins
    tests = iat_spec.get("tests")
    if tests:
        out = []
        for t in tests:
            tid = str(t)
            if tid not in stimuli:
                raise KeyError(f"IAT test_id '{tid}' not found in stimuli_json.")
            out.append(tid)
        return out

    include_categories = set(str(x) for x in (iat_spec.get("include_categories") or []))
    exclude_categories = set(str(x) for x in (iat_spec.get("exclude_categories") or []))

    out = []
    for test_id, d in stimuli.items():
        cat = str(d.get("category", ""))
        if include_categories and cat not in include_categories:
            continue
        if exclude_categories and cat in exclude_categories:
            continue
        out.append(str(test_id))

    out.sort()
    return out


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _already_done_keys(runs_csv: Path) -> set:
    """
    For resume: returns a set of tuples identifying completed IAT episodes.
    Key matches what run_iat_once records:
      (prompt_variant, model_key, seed, temperature, top_p, persona_label, system_prompt,
       test_id, order_id, task_seed, stimuli_sha256)
    Skips corrupt rows rather than raising.
    """
    if not runs_csv.exists():
        return set()
    try:
        df = pd.read_csv(runs_csv, on_bad_lines="skip")
    except Exception as e:
        print(f"[resume] WARNING: could not read {runs_csv}: {e} — treating as empty")
        return set()

    needed = {
        "prompt_variant", "model_key", "seed", "temperature", "top_p",
        "persona_label", "system_prompt", "test_id", "order_id", "task_seed", "stimuli_sha256",
    }
    if not needed.issubset(df.columns):
        print(f"[resume] WARNING: {runs_csv} missing columns {needed - set(df.columns)} — treating as empty")
        return set()

    df["system_prompt"] = df["system_prompt"].fillna("").astype(str)

    keys = set()
    n_skipped = 0
    for row in df.itertuples(index=False):
        try:
            keys.add((
                str(row.prompt_variant),
                str(row.model_key),
                int(float(row.seed)),
                float(row.temperature),
                float(row.top_p),
                str(row.persona_label),
                str(row.system_prompt),
                str(row.test_id),
                int(float(row.order_id)),
                int(float(row.task_seed)),
                str(row.stimuli_sha256),
            ))
        except (ValueError, TypeError):
            n_skipped += 1
    if n_skipped:
        print(f"[resume] WARNING: skipped {n_skipped} corrupt rows in {runs_csv}")
    return keys


# ---------------------------
# Optional: mock client for sweep smoke tests
# ---------------------------

class _DummyResp:
    def __init__(self, text: str):
        self.text = text


class _DummyLLM:
    def __init__(self, mode: str = "random"):
        self.mode = mode

    def chat(self, *, messages: List[Dict[str, str]], gen: GenParams) -> _DummyResp:
        user = messages[-1]["content"]
        m = re.search(r'assign one of "([^"]+)" or "([^"]+)"', user, re.IGNORECASE)
        sa = m.group(1) if m else "GroupA"
        sb = m.group(2) if m else "GroupB"

        m2 = re.search(r"The words are:\s*(.*)", user)
        words: List[str] = []
        if m2:
            wline = m2.group(1).splitlines()[0]
            words = [w.strip() for w in wline.split(",") if w.strip()]

        import random
        rng = random.Random(abs(hash((gen.seed, self.mode))) % (2**32))
        out = []
        for w in words:
            if self.mode == "all_sa":
                out.append(f"{w} - {sa}")
            elif self.mode == "all_sb":
                out.append(f"{w} - {sb}")
            else:
                out.append(f"{w} - {rng.choice([sa, sb])}")
        return _DummyResp("\n".join(out))


# ---------------------------
# Main
# ---------------------------

def _load_persona_prompts_from_selected(
    selected_path: str,
    *,
    config_path: str,
    seed: int,
    field: str = "persona_prompts",
    limit = None,
    prepend = None,
) -> List[str]:
    """Load PersonaHub persona prompts from selected_diverse_personas.json."""
    from pathlib import Path as _Path
    import json as _json
    for candidate in [
        _Path(selected_path),
        _Path(config_path).resolve().parent / selected_path,
        _Path.cwd() / selected_path,
    ]:
        if candidate.exists():
            p = candidate
            break
    else:
        raise FileNotFoundError(f"Could not resolve persona path: {selected_path}")

    with open(p, "r", encoding="utf-8") as f:
        sel = _json.load(f)

    key = f"seed_{int(seed)}"
    if "selections" not in sel or key not in sel["selections"]:
        raise KeyError(f"{p} missing selections.{key}")
    block = sel["selections"][key]
    if field not in block:
        raise KeyError(f"{p} selections.{key} missing field '{field}'")

    prompts = list(block[field] or [])
    if limit is not None:
        prompts = prompts[:int(limit)]
    out: List[str] = list(prepend or [])
    out.extend(prompts)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to IAT sweep spec JSON.")
    p.add_argument("--out_root", required=True, help="Root output directory, e.g., results/iat_sweeps")
    p.add_argument("--base_url", default=None, help="Override base URL for provider (optional).")
    p.add_argument("--skip_variant_path", action="store_true", help="Skip /exp_name/variant_id subpath (write directly into out_root)")

    # OpenRouter headers (optional)
    p.add_argument("--openrouter_referer", default=os.environ.get("OPENROUTER_REFERER", ""))
    p.add_argument("--openrouter_title", default=os.environ.get("OPENROUTER_TITLE", "psyai_eval"))

    # Behavior
    p.add_argument("--resume", action="store_true", help="Skip runs already present in iat_runs.csv")
    p.add_argument("--fail_fast", action="store_true", help="Stop on first error (default: continue)")

    # Testing / no-API mode
    p.add_argument("--mock_llm", choices=["none", "random", "all_sa", "all_sb"], default="none",
                   help="If set (not 'none'), run the sweep with a dummy LLM and no API calls.")

    args = p.parse_args()

    cfg = _load_json(args.config)
    exp_name = cfg.get("experiment_name", "iat_prompt_sweep")
    provider = str(cfg.get("provider", "openrouter")).lower().strip()
    models_config_path = str(cfg.get("models_config", "configs/openrouter_models.json"))
    model_keys: Optional[List[str]] = cfg.get("model_keys") or None

    # grid
    grid = cfg.get("grid", {})
    seeds: List[int] = list(grid.get("seeds", [42, 99, 123]))
    temperatures: List[float] = list(grid.get("temperatures", [0.7, 0.9]))
    top_ps: List[float] = list(grid.get("top_ps", [0.95, 1.0]))
    persona_mode = str(grid.get("persona_prompts_mode", "direct")).strip().lower()
    if persona_mode == "selected_personas":
        sel = grid.get("persona_prompts_selected", {}) or {}
        persona_prompts: List[str] = _load_persona_prompts_from_selected(
            str(sel.get("path", "configs/selected_diverse_personas.json")),
            config_path=args.config,
            seed=int(sel.get("seed", 42)),
            field=str(sel.get("field", "persona_prompts")),
            limit=sel.get("limit", None),
            prepend=sel.get("prepend", None),
        )
        print(f"[personas] loaded {len(persona_prompts)} personas from selected_personas (seed_{sel.get('seed', 42)})")
    else:
        persona_prompts: List[str] = list(grid.get("persona_prompts", [""]))
    task_seed: int = int(grid.get("task_seed", cfg.get("task_seed", 123)))

    max_tokens: int = int(cfg.get("max_tokens", 900))
    store_raw_response: bool = bool(cfg.get("store_raw_response", False))

    variants = cfg.get("variants", [])
    if not variants:
        raise ValueError("Spec must include a non-empty 'variants' list.")

    # IAT spec
    iat_spec = cfg.get("iat", {}) or {}
    stimuli_json = str(iat_spec.get("stimuli_json", "")).strip()
    if not stimuli_json:
        raise ValueError("Spec must include iat.stimuli_json")
    stimuli_sha = _file_sha256(stimuli_json)

    stimuli = IATEnv.load_stimuli(stimuli_json)
    tests = _select_tests(stimuli, iat_spec)

    base_iat_cfg = _apply_iat_overrides(IATConfig(), iat_spec)

    models_cfg = _load_json(models_config_path)
    model_pairs = _models_from_config(models_cfg, provider, model_keys)

    out_root = Path(args.out_root) / _safe_slug(str(exp_name))
    out_root.mkdir(parents=True, exist_ok=True)

    llm_cache: Dict[str, OpenAICompatChatClient] = {}
    headers = {"HTTP-Referer": args.openrouter_referer, "X-Title": args.openrouter_title}

    for v in variants:
        variant_id = str(v.get("variant_id", "")).strip()
        if not variant_id:
            raise ValueError("Each variant must have a non-empty variant_id.")

        tact = _tact_from_dict(v.get("tact", {}))

        base_system_prompt = str(v.get("base_system_prompt", "") or "")
        task_context = str(v.get("task_context", "") or "")
        extra_system_prompt = str(v.get("system_prompt", "") or "")

        iat_cfg = _apply_iat_overrides(base_iat_cfg, v.get("iat", {}))

        v_dir = out_root / _safe_slug(variant_id)
        if args.skip_variant_path:
            v_dir = Path(args.out_root)
        v_dir.mkdir(parents=True, exist_ok=True)

        (v_dir / "variant_effective.json").write_text(
            json.dumps(
                {
                    "variant_id": variant_id,
                    "provider": provider,
                    "models_config": models_config_path,
                    "model_keys": [k for k, _ in model_pairs],
                    "grid": {
                        "seeds": seeds,
                        "temperatures": temperatures,
                        "top_ps": top_ps,
                        "persona_prompts": persona_prompts,
                        "task_seed": task_seed,
                    },
                    "max_tokens": max_tokens,
                    "store_raw_response": store_raw_response,
                    "prompts": {
                        "base_system_prompt": base_system_prompt,
                        "task_context": task_context,
                        "extra_system_prompt": extra_system_prompt,
                    },
                    "tact": {
                        "target": tact.target,
                        "action": tact.action,
                        "context": tact.context,
                        "time": tact.time,
                        "policy_label": tact.policy_label,
                    },
                    "iat": {
                        "stimuli_json": stimuli_json,
                        "stimuli_sha256": stimuli_sha,
                        "tests": tests,
                        "iat_config": {f.name: getattr(iat_cfg, f.name) for f in fields(IATConfig)},
                    },
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        out_steps_csv = v_dir / "iat_steps.csv"
        out_runs_csv = v_dir / "iat_runs.csv"
        if args.resume:
            try:
                done = _already_done_keys(out_runs_csv)
                print(f"[resume] variant={variant_id} — found {len(done)} completed keys in {out_runs_csv}")
            except Exception as e:
                print(f"[resume] WARNING: failed to load done_keys for variant={variant_id} ({e}) — will re-run all")
                done = set()
        else:
            done = set()

        episodes: List[Tuple[str, int]] = []
        for test_id in tests:
            for order_id in range(int(iat_cfg.orders_per_test)):
                episodes.append((test_id, order_id))

        total = (
            len(model_pairs)
            * len(seeds)
            * len(temperatures)
            * len(top_ps)
            * len(persona_prompts)
            * len(episodes)
        )
        bar = tqdm(total=total, desc=f"IAT sweep [{variant_id}]", ncols=110)

        for model_key, model_id in model_pairs:
            if args.mock_llm != "none":
                llm = _DummyLLM(mode=args.mock_llm)
                eff_provider = "mock"
            else:
                if model_id not in llm_cache:
                    llm_cache[model_id] = _make_client(provider, model_id, args.base_url, headers)
                llm = llm_cache[model_id]
                eff_provider = provider

            for seed in seeds:
                for t in temperatures:
                    for top_p in top_ps:
                        for persona in persona_prompts:
                            persona = persona or ""

                            sys_parts = [extra_system_prompt.strip(), persona.strip()]
                            system_prompt = "\n".join([s for s in sys_parts if s]).strip()

                            # FIXED: label based on final system_prompt (matches runner logs)
                            persona_lbl = _persona_label(system_prompt)

                            for test_id, order_id in episodes:
                                key = (
                                    variant_id,
                                    model_key,
                                    int(seed),
                                    float(t),
                                    float(top_p),
                                    persona_lbl,
                                    system_prompt,
                                    str(test_id),
                                    int(order_id),
                                    int(task_seed),
                                    stimuli_sha,
                                )
                                if key in done:
                                    bar.update(1)
                                    continue

                                bar.set_postfix(
                                    {
                                        "model": model_key,
                                        "seed": seed,
                                        "t": t,
                                        "p": top_p,
                                        "persona": persona_lbl,
                                        "test": str(test_id)[:16],
                                        "o": order_id,
                                    }
                                )

                                gen = GenParams(
                                    temperature=float(t),
                                    top_p=float(top_p),
                                    max_tokens=max_tokens,
                                    seed=int(seed),
                                )

                                try:
                                    run_iat_once(
                                        llm=llm,
                                        gen=gen,
                                        tact=tact,
                                        cfg=iat_cfg,
                                        task_seed=int(task_seed),
                                        out_steps_csv=str(out_steps_csv),
                                        out_runs_csv=str(out_runs_csv),
                                        prompt_variant=variant_id,
                                        base_system_prompt=base_system_prompt,
                                        task_context=task_context,
                                        system_prompt=system_prompt,
                                        store_raw_response=store_raw_response,
                                        model_key=model_key,
                                        model_id=model_id,
                                        provider=eff_provider,
                                        stimuli_json=stimuli_json,
                                        test_id=str(test_id),
                                        order_id=int(order_id),
                                    )
                                except Exception as e:
                                    if args.fail_fast:
                                        raise
                                    err_row = pd.DataFrame(
                                        [
                                            {
                                                "prompt_variant": variant_id,
                                                "model_key": model_key,
                                                "model_id": model_id,
                                                "seed": int(seed),
                                                "temperature": float(t),
                                                "top_p": float(top_p),
                                                "persona_label": persona_lbl,
                                                "system_prompt": system_prompt,
                                                "task_seed": int(task_seed),
                                                "stimuli_json": stimuli_json,
                                                "stimuli_sha256": stimuli_sha,
                                                "test_id": str(test_id),
                                                "order_id": int(order_id),
                                                "error": repr(e),
                                            }
                                        ]
                                    )
                                    header = not out_runs_csv.exists()
                                    err_row.to_csv(out_runs_csv, mode="a", header=header, index=False)
                                finally:
                                    bar.update(1)

        bar.close()

    print(f"Done. Outputs under: {out_root}")


if __name__ == "__main__":
    main()