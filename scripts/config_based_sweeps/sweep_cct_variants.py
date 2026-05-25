#!/usr/bin/env python3
"""
Run a prompt sweep for the Columbia Card Task (CCT) using a JSON spec, mirroring TPB sweeps.

Expected spec format (example):
{
  "experiment_name": "cct_prompt_sweep_v1",
  "provider": "openrouter",
  "models_config": "configs/openrouter_models.json",
  "model_keys": ["llama31_8b","qwen_72b","gpt4o_mini"],
  "grid": {
    "seeds": [42,99,123],
    "temperatures": [0.7,0.9],
    "top_ps": [0.95,1.0],
    "persona_prompts": ["", "You are a helpful assistant.", "Respond to instructions."],
    "task_seed": 123
  },
  "max_tokens": 700,
  "store_raw_response": false,
  "allow_nonjson": false,
  "cct": { "n_rounds": 10, "max_flips": 16, ... },   # optional env overrides
  "variants": [
    {
      "variant_id": "balanced_no_numbers",
      "base_system_prompt": "...",
      "task_context": "...",
      "system_prompt": "",              # optional extra system prompt
      "tact": { "target":"I", ... },    # required
      "cct": { ... }                    # optional per-variant env overrides
    }
  ]
}

Outputs:
  <out_root>/<experiment_name>/cct_sweep/<variant_id>/cct_steps.csv
  <out_root>/<experiment_name>/cct_sweep/<variant_id>/cct_runs.csv
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
from psyai_eval.runner.cct_runner import run_cct_once
from psyai_eval.tasks.cct.env import CCTConfig


# ---------------------------
# Helpers: config parsing
# ---------------------------

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_slug(s: str, max_len: int = 60) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:max_len] if len(s) > max_len else s


def _persona_label(persona: str) -> str:
    p = (persona or "").strip()
    if p == "":
        return "empty"
    if p == "You are a helpful assistant.":
        return "helpful"
    if p == "Respond to instructions.":
        return "instructions"
    # Use SHA-256 for a deterministic label stable across Python processes.
    # (Python's built-in hash() is randomised per-process via PYTHONHASHSEED.)
    digest = hashlib.sha256(p.encode("utf-8")).hexdigest()
    return f"p{digest[:4]}"


def _resolve_path(p: str, *, config_path: str) -> Path:
    """
    Resolve a possibly-relative path:
      1) as given (absolute or relative to cwd)
      2) relative to config file directory
      3) relative to current working directory (explicit fallback)
    """
    cand = Path(p)
    if cand.exists():
        return cand
    cand = Path(config_path).resolve().parent / p
    if cand.exists():
        return cand
    cand = Path.cwd() / p
    if cand.exists():
        return cand
    raise FileNotFoundError(f"Could not resolve path: {p!r} (config={config_path})")


def _load_persona_prompts_from_selected(
    selected_path: str,
    *,
    config_path: str,
    seed: int,
    field: str = "persona_prompts",
    limit: Optional[int] = None,
    prepend: Optional[List[str]] = None,
) -> List[str]:
    """
    Load vignette/persona prompts from a selected_personas JSON file
    (e.g. selected_vignettes_v1.json).

    Expected structure:
      { "selections": { "seed_1": { "persona_prompts": [...], ... }, ... } }

    This mirrors the same helper in sweep_tpb_variants.py so both sweeps
    load vignettes from the same file with identical logic.
    """
    p = _resolve_path(selected_path, config_path=config_path)
    with open(p, "r", encoding="utf-8") as f:
        sel = json.load(f)

    key = f"seed_{int(seed)}"
    if "selections" not in sel or key not in sel["selections"]:
        raise KeyError(
            f"{p} missing selections.{key}. "
            f"Available keys: {list(sel.get('selections', {}).keys())}"
        )

    block = sel["selections"][key]
    if field not in block:
        raise KeyError(
            f"{p} selections.{key} missing field '{field}'. "
            f"Available fields: {list(block.keys())}"
        )

    prompts = list(block[field] or [])
    if limit is not None:
        prompts = prompts[: int(limit)]

    out: List[str] = []
    if prepend:
        out.extend(prepend)
    out.extend(prompts)
    return out


def _models_from_config(models_cfg: dict, provider: str, model_keys: Optional[List[str]]) -> List[Tuple[str, str]]:
    """
    Returns [(model_key, model_id), ...] for this provider.
    Matches the existing configs/openrouter_models.json format:
      { "openrouter": { "llama31_8b": "...", ... }, "llama": {...} }
    """
    provider = provider.lower().strip()
    if provider not in models_cfg:
        raise ValueError(f"Provider '{provider}' not found in models config. Available: {list(models_cfg.keys())}")

    block = models_cfg[provider]
    if not isinstance(block, dict):
        raise ValueError(f"Invalid models config format for provider '{provider}'. Must be dict of key->model_id.")

    if model_keys:
        missing = [k for k in model_keys if k not in block]
        if missing:
            raise ValueError(f"Requested model_keys not in config: {missing}. Available: {list(block.keys())}")
        return [(k, str(block[k])) for k in model_keys]

    return [(k, str(v)) for k, v in block.items()]


# ---------------------------
# Helpers: provider clients
# ---------------------------

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
# Helpers: CCT env config
# ---------------------------

def _filter_dataclass_kwargs(dc_type, d: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {f.name for f in fields(dc_type)}
    return {k: v for k, v in (d or {}).items() if k in allowed}


def _apply_cct_overrides(base_cfg: CCTConfig, overrides: Dict[str, Any]) -> CCTConfig:
    """
    CCTConfig is a frozen dataclass, so we must return a new instance via dataclasses.replace.
    Unknown keys are ignored (prevents crashes if the JSON spec includes extra fields).
    """
    kw = _filter_dataclass_kwargs(CCTConfig, overrides or {})
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


def _already_done_keys(runs_csv: Path) -> set:
    """
    For resume: returns a set of tuples identifying completed runs.
    Key matches what run_cct_once records: (model_key, seed, temperature, top_p, system_prompt)
    plus prompt_variant to separate variants. persona_label is intentionally excluded —
    system_prompt already encodes the full persona content and is stable across runs.
    Skips corrupt rows rather than raising.
    """
    if not runs_csv.exists():
        return set()
    try:
        df = pd.read_csv(runs_csv, on_bad_lines="skip")
    except Exception as e:
        print(f"[resume] WARNING: could not read {runs_csv}: {e} — treating as empty")
        return set()

    needed = {"prompt_variant", "model_key", "seed", "temperature", "top_p", "system_prompt"}
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
                str(row.system_prompt),
            ))
        except (ValueError, TypeError):
            n_skipped += 1
    if n_skipped:
        print(f"[resume] WARNING: skipped {n_skipped} corrupt rows in {runs_csv}")
    return keys


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to CCT sweep spec JSON.")
    p.add_argument("--out_root", required=True, help="Root output directory, e.g., results/cct_sweeps")
    p.add_argument("--base_url", default=None, help="Override base URL for provider (optional).")
    p.add_argument("--skip_variant_path", action="store_true", help="If used, skips putting outputs in the /exp_name/variant_id subpath")

    # OpenRouter headers (optional)
    p.add_argument("--openrouter_referer", default=os.environ.get("OPENROUTER_REFERER", ""))
    p.add_argument("--openrouter_title", default=os.environ.get("OPENROUTER_TITLE", "psyai_eval"))

    # Behavior
    p.add_argument("--resume", action="store_true", help="Skip runs already present in cct_runs.csv")
    p.add_argument("--fail_fast", action="store_true", help="Stop on first error (default: continue)")

    args = p.parse_args()

    cfg = _load_json(args.config)
    exp_name = cfg.get("experiment_name", "cct_prompt_sweep")
    provider = str(cfg.get("provider", "openrouter")).lower().strip()
    models_config_path = str(cfg.get("models_config", "configs/openrouter_models.json"))
    model_keys: Optional[List[str]] = cfg.get("model_keys") or None

    # grid
    grid = cfg.get("grid", {})
    seeds: List[int] = list(grid.get("seeds", [42, 99, 123]))
    temperatures: List[float] = list(grid.get("temperatures", [0.7, 0.9]))
    top_ps: List[float] = list(grid.get("top_ps", [0.95, 1.0]))
    task_seed: int = int(grid.get("task_seed", cfg.get("task_seed", 123)))

    # Vignette / persona loading — mirrors sweep_tpb_variants.py exactly.
    # persona_prompts_mode="direct"            → read grid.persona_prompts list (legacy / v3)
    # persona_prompts_mode="selected_personas" → load from external JSON file (v4 vignette design)
    persona_mode = str(grid.get("persona_prompts_mode", "direct")).strip().lower()
    if persona_mode not in ("direct", "selected_personas"):
        raise ValueError("grid.persona_prompts_mode must be 'direct' or 'selected_personas'")

    if persona_mode == "selected_personas":
        sel = grid.get("persona_prompts_selected", {}) or {}
        selected_path = str(sel.get("path", "configs/selected_vignettes_v1.json"))
        selected_seed = int(sel.get("seed", 1))
        selected_field = str(sel.get("field", "persona_prompts"))
        selected_limit = sel.get("limit", None)
        selected_prepend = sel.get("prepend", None)
        persona_prompts: List[str] = _load_persona_prompts_from_selected(
            selected_path,
            config_path=args.config,
            seed=selected_seed,
            field=selected_field,
            limit=selected_limit,
            prepend=selected_prepend,
        )
        print(f"[personas] loaded {len(persona_prompts)} vignettes from {selected_path} (seed_{selected_seed})")
    else:
        persona_prompts = list(grid.get("persona_prompts", [""]))

    max_tokens: int = int(cfg.get("max_tokens", 700))
    store_raw_response: bool = bool(cfg.get("store_raw_response", False))
    allow_nonjson: bool = bool(cfg.get("allow_nonjson", False))

    variants = [v for v in cfg.get("variants", []) if v.get("enabled", True)]
    if not variants:
        raise ValueError("Spec must include a non-empty 'variants' list.")

    models_cfg = _load_json(models_config_path)
    model_pairs = _models_from_config(models_cfg, provider, model_keys)

    # Base (frozen) env config: start from defaults then apply spec-level overrides
    base_env_cfg = CCTConfig()
    base_env_cfg = _apply_cct_overrides(base_env_cfg, cfg.get("cct", {}))

    out_root = Path(args.out_root) / _safe_slug(str(exp_name))
    out_root.mkdir(parents=True, exist_ok=True)

    # cache LLM clients per model_id
    llm_cache: Dict[str, OpenAICompatChatClient] = {}
    headers = {
        "HTTP-Referer": args.openrouter_referer,
        "X-Title": args.openrouter_title,
    }

    for v in variants:
        variant_id = str(v.get("variant_id", "")).strip()
        if not variant_id:
            raise ValueError("Each variant must have a non-empty variant_id.")

        tact = _tact_from_dict(v.get("tact", {}))

        # Variant-level persona_prompts override grid-level (used by steering configs).
        # Falls back to grid persona_prompts for baseline/neutral configs.
        #persona_prompts: List[str] = list(v.get("persona_prompts") or grid.get("persona_prompts", [""]))
        effective_persona_prompts: List[str] = list(v.get("persona_prompts") or persona_prompts)

        # prompts
        base_system_prompt = str(v.get("base_system_prompt", "") or "")
        task_context = str(v.get("task_context", "") or "")
        extra_system_prompt = str(v.get("system_prompt", "") or "")

        # env cfg overrides per variant
        env_cfg = _apply_cct_overrides(base_env_cfg, v.get("cct", {}))

        v_dir = out_root / _safe_slug(variant_id)
        if args.skip_variant_path:
            v_dir = Path(args.out_root)
        v_dir.mkdir(parents=True, exist_ok=True)

        # Write the effective variant spec for provenance
        (v_dir / "variant_effective.json").write_text(
            json.dumps(
                {
                    "variant_id": variant_id,
                    "provider": provider,
                    "models_config": models_config_path,
                    "model_keys": [k for k, _ in model_pairs],
                    "grid": {"seeds": seeds, "temperatures": temperatures, "top_ps": top_ps, "persona_prompts": effective_persona_prompts, "task_seed": task_seed},  # persona_prompts is variant-resolved
                    "max_tokens": max_tokens,
                    "store_raw_response": store_raw_response,
                    "allow_nonjson": allow_nonjson,
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
                    "cct_env": {f.name: getattr(env_cfg, f.name) for f in fields(CCTConfig)},
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        out_steps_csv = v_dir / "cct_steps.csv"
        out_runs_csv = v_dir / "cct_runs.csv"

        if args.resume:
            try:
                done = _already_done_keys(out_runs_csv)
                print(f"[resume] variant={variant_id} — found {len(done)} completed keys in {out_runs_csv}")
            except Exception as e:
                print(f"[resume] WARNING: failed to load done_keys for variant={variant_id} ({e}) — will re-run all")
                done = set()
        else:
            done = set()

        #total = len(model_pairs) * len(seeds) * len(temperatures) * len(top_ps) * len(persona_prompts)
        total = len(model_pairs) * len(seeds) * len(temperatures) * len(top_ps) * len(effective_persona_prompts)
        bar = tqdm(total=total, desc=f"CCT sweep [{variant_id}]", ncols=110)

        for model_key, model_id in model_pairs:
            if model_id not in llm_cache:
                llm_cache[model_id] = _make_client(provider, model_id, args.base_url, headers)
            llm = llm_cache[model_id]

            for seed in seeds:
                for t in temperatures:
                    for top_p in top_ps:
                        for persona in effective_persona_prompts:
                            persona = persona or ""
                            persona_lbl = _persona_label(persona)

                            # system_prompt = extra_system_prompt + persona prompt (both as system content)
                            sys_parts = [extra_system_prompt.strip(), persona.strip()]
                            system_prompt = "\n".join([s for s in sys_parts if s]).strip()

                            key = (variant_id, model_key, int(seed), float(t), float(top_p), system_prompt)
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
                                }
                            )

                            gen = GenParams(
                                temperature=float(t),
                                top_p=float(top_p),
                                max_tokens=max_tokens,
                                seed=int(seed),
                            )

                            try:
                                run_cct_once(
                                    llm=llm,
                                    gen=gen,
                                    tact=tact,
                                    cfg=env_cfg,
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
                                    provider=provider,
                                    steering=v.get("steering"),
                                    #allow_nonjson=allow_nonjson,
                                    #extra_run_fields={
                                    #    "experiment_name": exp_name,
                                    #    "variant_id": variant_id,
                                    #    "models_config": models_config_path,
                                    #},
                                )
                            except Exception as e:
                                if args.fail_fast:
                                    raise
                                # Record the failure minimally into runs CSV so the sweep can continue.
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