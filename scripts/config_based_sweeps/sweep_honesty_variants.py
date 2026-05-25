#!/usr/bin/env python3
"""
Run a prompt sweep for the Honesty behavior sandbox using a JSON spec (mirrors CCT/IAT sweeps).

Spec format (example):
{
  "experiment_name": "honesty_prompt_sweep_v1",
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
  "max_tokens": 64,
  "store_raw_response": false,
  "honesty": {
    "questions_csv": "configs/honesty/norm300_syn.csv",
    "questions_per_bin": 10,
    "question_ids": null
  },
  "variants": [
    {
      "variant_id": "neutral_honesty",
      "base_system_prompt": "",
      "task_context": "",
      "system_prompt": "",          // optional extra system prompt
      "tact": { ... },              // required
      "honesty": { ... }            // optional per-variant HonestyConfig overrides
    }
  ]
}

Outputs:
  <out_root>/<experiment_name>/<variant_id>/honesty_steps.csv
  <out_root>/<experiment_name>/<variant_id>/honesty_runs.csv
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
from psyai_eval.runner.honesty_runner import run_honesty_once
from psyai_eval.tasks.honesty.env import HonestyConfig, HonestyEnv


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
    # IMPORTANT: must match runner logic
    p = (system_prompt or "").strip()
    if p == "":
        return "empty"
    if p == "You are a helpful assistant.":
        return "helpful"
    if p == "Respond to instructions.":
        return "instructions"
    return "p" + hashlib.sha256(p.encode("utf-8")).hexdigest()[:4]


def _models_from_config(models_cfg: dict, provider: str, model_keys: Optional[List[str]]) -> List[Tuple[str, str]]:
    provider = provider.lower().strip()
    if provider not in models_cfg:
        raise ValueError(f"Provider '{provider}' not found in models config. Available: {list(models_cfg.keys())}")
    mp = models_cfg[provider]
    if not isinstance(mp, dict):
        raise ValueError(f"models_config['{provider}'] must map model_key -> model_id")

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
# Honesty env config helpers
# ---------------------------

def _filter_dataclass_kwargs(dc_type, d: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {f.name for f in fields(dc_type)}
    return {k: v for k, v in (d or {}).items() if k in allowed}


def _apply_honesty_overrides(base_cfg: HonestyConfig, overrides: Dict[str, Any]) -> HonestyConfig:
    kw = _filter_dataclass_kwargs(HonestyConfig, overrides or {})
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


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_persona_prompts_from_selected(
    selected_path: str,
    *,
    config_path: str,
    seed: int,
    field: str = "persona_prompts",
    limit: Optional[int] = None,
    prepend: Optional[List[str]] = None,
) -> List[str]:
    """Load PersonaHub persona prompts from selected_diverse_personas.json."""
    # Resolve path relative to config file, then CWD
    for candidate in [
        Path(selected_path),
        Path(config_path).resolve().parent / selected_path,
        Path.cwd() / selected_path,
    ]:
        if candidate.exists():
            p = candidate
            break
    else:
        raise FileNotFoundError(f"Could not resolve persona path: {selected_path} (config={config_path})")

    with open(p, "r", encoding="utf-8") as f:
        sel = json.load(f)

    key = f"seed_{int(seed)}"
    if "selections" not in sel or key not in sel["selections"]:
        raise KeyError(f"{p} missing selections.{key}")

    block = sel["selections"][key]
    if field not in block:
        raise KeyError(f"{p} selections.{key} missing field '{field}'")

    prompts = list(block[field] or [])
    if limit is not None:
        prompts = prompts[:int(limit)]

    out: List[str] = []
    if prepend:
        out.extend(prepend)
    out.extend(prompts)
    return out


def _already_done_keys(runs_csv: Path) -> set:
    """
    For resume: returns a set of tuples identifying completed Honesty episodes.

    Prefer stable question identity: (difficulty_bin, original_item_number) rather than question_id.
    Skips corrupt rows (e.g. from malformed CSV appends) rather than raising.
    """
    if not runs_csv.exists():
        return set()
    try:
        df = pd.read_csv(runs_csv, on_bad_lines="skip")
    except Exception as e:
        print(f"[resume] WARNING: could not read {runs_csv}: {e} — treating as empty")
        return set()

    needed = {
        "prompt_variant",
        "model_key",
        "seed",
        "temperature",
        "top_p",
        "persona_label",
        "system_prompt",
        "task_seed",
        "questions_sha256",
        "difficulty_bin",
        "original_item_number",
    }
    if not needed.issubset(df.columns):
        print(f"[resume] WARNING: {runs_csv} missing columns {needed - set(df.columns)} — treating as empty")
        return set()

    # Normalize NaN system_prompt to "" to match what the main loop produces
    df["system_prompt"] = df["system_prompt"].fillna("").astype(str)

    keys = set()
    n_skipped = 0
    for row in df.itertuples(index=False):
        try:
            key = (
                str(row.prompt_variant),
                str(row.model_key),
                int(float(row.seed)),
                float(row.temperature),
                float(row.top_p),
                str(row.persona_label),
                str(row.system_prompt),
                int(float(row.task_seed)),
                str(row.questions_sha256),
                int(float(row.difficulty_bin)) if str(row.difficulty_bin) not in ("", "nan") else -1,
                int(float(row.original_item_number)) if str(row.original_item_number) not in ("", "nan") else -1,
            )
            keys.add(key)
        except (ValueError, TypeError):
            n_skipped += 1
            continue
    if n_skipped:
        print(f"[resume] WARNING: skipped {n_skipped} corrupt rows in {runs_csv}")
    return keys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to honesty sweep config JSON.")
    ap.add_argument("--out_root", default="results/honesty_sweeps", help="Output root directory.")
    ap.add_argument("--resume", action="store_true", help="Skip already-completed episodes if runs CSV exists.")
    ap.add_argument("--fail_fast", action="store_true", help="Stop on first error.")
    ap.add_argument("--model_keys", nargs="+", default=None,
                    help="Optional subset of model keys to run (overrides config list).")
    args = ap.parse_args()

    cfg = _load_json(args.config)

    exp_name = str(cfg.get("experiment_name", "honesty_prompt_sweep")).strip()
    provider = str(cfg.get("provider", "openrouter")).strip()
    models_cfg_path = str(cfg.get("models_config", "configs/openrouter_models.json")).strip()
    model_keys = args.model_keys or cfg.get("model_keys") or None

    grid = cfg.get("grid") or {}
    seeds = list(grid.get("seeds") or [42])
    temperatures = list(grid.get("temperatures") or [0.0])
    top_ps = list(grid.get("top_ps") or [1.0])
    persona_mode = str(grid.get("persona_prompts_mode", "direct")).strip().lower()
    if persona_mode == "selected_personas":
        sel = grid.get("persona_prompts_selected", {}) or {}
        persona_prompts = _load_persona_prompts_from_selected(
            str(sel.get("path", "configs/selected_diverse_personas.json")),
            config_path=args.config,
            seed=int(sel.get("seed", 42)),
            field=str(sel.get("field", "persona_prompts")),
            limit=sel.get("limit", None),
            prepend=sel.get("prepend", None),
        )
        print(f"[personas] loaded {len(persona_prompts)} personas from selected_personas (seed_{sel.get('seed', 42)})")
    else:
        persona_prompts = list(grid.get("persona_prompts") or [""])
    task_seed = int(grid.get("task_seed", 123))

    max_tokens = int(cfg.get("max_tokens", 64))
    store_raw_response = bool(cfg.get("store_raw_response", False))

    headers = cfg.get("headers") or {}
    base_url = cfg.get("base_url") or None

    honesty_spec = cfg.get("honesty") or {}
    questions_csv = str(honesty_spec.get("questions_csv", "")).strip()
    questions_per_bin = int(honesty_spec.get("questions_per_bin", 10))
    question_ids = honesty_spec.get("question_ids")  # optional list[int]
    if questions_csv == "":
        raise ValueError("config.honesty.questions_csv is required")

    questions_sha = _file_sha256(questions_csv)
    questions = HonestyEnv.load_questions(questions_csv, questions_per_bin=questions_per_bin)

    if question_ids:
        wanted = set(int(x) for x in question_ids)
        questions = [q for q in questions if int(q.id) in wanted]
        if not questions:
            raise ValueError("After filtering by question_ids, no questions remain.")

    models_cfg = _load_json(models_cfg_path)
    model_pairs = _models_from_config(models_cfg, provider, model_keys)

    variants = cfg.get("variants") or []
    if not isinstance(variants, list) or not variants:
        raise ValueError("config.variants must be a non-empty list")

    out_root = Path(args.out_root) / exp_name
    out_root.mkdir(parents=True, exist_ok=True)

    # Expand plan
    plan = []
    for v in variants:
        variant_id = str(v.get("variant_id", "")).strip()
        if not variant_id:
            raise ValueError("Each variant must have a non-empty variant_id")
        for (model_key, model_id) in model_pairs:
            for seed in seeds:
                for temp in temperatures:
                    for top_p in top_ps:
                        for persona in persona_prompts:
                            plan.append((variant_id, v, model_key, model_id, int(seed), float(temp), float(top_p), str(persona)))

    # Load done_keys ONCE before the loop — not per-condition
    # Also pre-compute per-condition skip check to filter plan accurately
    variant_dirs = {}
    for v in variants:
        vid = _safe_slug(str(v.get("variant_id", "")))
        variant_dirs[str(v.get("variant_id", ""))] = out_root / vid

    if args.resume:
        # Load done_keys once per variant
        done_keys_by_variant: dict = {}
        for v in variants:
            vid = str(v.get("variant_id", ""))
            vdir = variant_dirs[vid]
            runs_csv = vdir / "honesty_runs.csv"
            try:
                dk = _already_done_keys(runs_csv)
                done_keys_by_variant[vid] = dk
                print(f"[resume] variant={vid} — found {len(dk)} completed question keys in {runs_csv}")
            except Exception as e:
                print(f"[resume] WARNING: failed to load done_keys for variant={vid} ({e}) — will re-run all")
                done_keys_by_variant[vid] = set()

        # Pre-filter plan: skip conditions where all questions are already done
        def _condition_all_done(variant_id, model_key, seed, temp, top_p, persona_prompt,
                                system_prompt, task_seed, questions_sha, questions, done_keys):
            persona_label = _persona_label(system_prompt)
            for q in questions:
                q_bin  = int(q.difficulty_bin) if q.difficulty_bin is not None else -1
                q_item = int(q.original_item_number) if q.original_item_number is not None else -1
                key = (str(variant_id), str(model_key), int(seed), float(temp), float(top_p),
                       str(persona_label), str(system_prompt), int(task_seed),
                       str(questions_sha), int(q_bin), int(q_item))
                if key not in done_keys:
                    return False
            return True

        filtered_plan = []
        _debug_shown = False
        for item in plan:
            vid, v, mk, mid, seed, temp, top_p, persona_prompt = item
            extra_sys = v.get("system_prompt") or ""
            sys_prompt = (extra_sys.strip() + "\n\n" + persona_prompt.strip()).strip() if extra_sys and persona_prompt else (extra_sys.strip() or persona_prompt.strip())
            dk = done_keys_by_variant.get(vid, set())
            # Honesty runner writes system_prompt=NaN to CSV; _already_done_keys
            # normalises it to "". Always pass "" here to match stored keys.
            all_done = _condition_all_done(vid, mk, seed, temp, top_p, persona_prompt,
                                           "", task_seed, questions_sha, questions, dk)
            if not all_done:
                filtered_plan.append(item)
        n_skipped = len(plan) - len(filtered_plan)
        if n_skipped:
            print(f"[resume] skipping {n_skipped} fully-completed conditions, {len(filtered_plan)} remaining")
        plan = filtered_plan
    else:
        done_keys_by_variant = {str(v.get("variant_id","")): set() for v in variants}

    for (variant_id, v, model_key, model_id, seed, temp, top_p, persona_prompt) in tqdm(plan, desc=f"honesty sweep {exp_name}"):
        variant_dir = out_root / _safe_slug(variant_id)
        variant_dir.mkdir(parents=True, exist_ok=True)
        out_steps_csv = variant_dir / "honesty_steps.csv"
        out_runs_csv = variant_dir / "honesty_runs.csv"
        effective_json = variant_dir / "variant_effective.json"

        done_keys = done_keys_by_variant.get(variant_id, set())

        # Variant prompts
        base_system_prompt = v.get("base_system_prompt") or ""
        task_context = v.get("task_context") or ""
        extra_system_prompt = v.get("system_prompt") or ""

        # Per-run persona system prompt (CCT/IAT style: extra_system_prompt + persona_prompt)
        system_prompt = ""
        if extra_system_prompt and persona_prompt:
            system_prompt = extra_system_prompt.strip() + "\n\n" + persona_prompt.strip()
        elif extra_system_prompt:
            system_prompt = extra_system_prompt.strip()
        else:
            system_prompt = persona_prompt.strip()

        tact = _tact_from_dict(v.get("tact") or {})

        # HonestyConfig overrides (global + per-variant)
        base_env_cfg = HonestyConfig()
        global_env_overrides = (honesty_spec.get("env") or {})
        variant_env_overrides = (v.get("env") or v.get("honesty") or {})
        # If the variant uses "honesty": {...} for env overrides, accept it.
        env_cfg = _apply_honesty_overrides(_apply_honesty_overrides(base_env_cfg, global_env_overrides), variant_env_overrides)
# Persist effective config once
        if not effective_json.exists():
            eff = {
                "experiment_name": exp_name,
                "provider": provider,
                "model_key": model_key,
                "model_id": model_id,
                "questions_csv": questions_csv,
                "questions_sha256": questions_sha,
                "questions_per_bin": questions_per_bin,
                "honesty_env": {
                    "enforce_format_in_system": env_cfg.enforce_format_in_system,
                    "confidence_scale_max": env_cfg.confidence_scale_max,
                    "fuzzy_thresholds": list(env_cfg.fuzzy_thresholds),
                },
                "max_tokens": max_tokens,
                "store_raw_response": store_raw_response,
                "variant": {
                    "variant_id": variant_id,
                    "base_system_prompt": base_system_prompt,
                    "task_context": task_context,
                    "system_prompt": extra_system_prompt,
                    "tact": {
                        "target": tact.target,
                        "action": tact.action,
                        "context": tact.context,
                        "time": tact.time,
                        "policy_label": tact.policy_label,
                    },
                },
            }
            effective_json.write_text(json.dumps(eff, indent=2), encoding="utf-8")

        # Client
        llm = _make_client(provider=provider, model=model_id, base_url=base_url, headers=headers)

        gen = GenParams(temperature=temp, top_p=top_p, max_tokens=max_tokens, seed=seed)

        for q in questions:
            # build resume key
            persona_label = _persona_label(system_prompt)
            q_bin = int(q.difficulty_bin) if q.difficulty_bin is not None else -1
            q_item = int(q.original_item_number) if q.original_item_number is not None else -1

            key = (
                str(variant_id),
                str(model_key),
                int(seed),
                float(temp),
                float(top_p),
                str(persona_label),
                str(system_prompt),
                int(task_seed),
                str(questions_sha),
                int(q_bin),
                int(q_item),
            )
            if args.resume and key in done_keys:
                continue

            try:
                run_honesty_once(
                    llm=llm,
                    questions=questions,
                    question_id=int(q.id),
                    out_dir=str(variant_dir),
                    out_steps_csv=str(out_steps_csv),
                    out_runs_csv=str(out_runs_csv),
                    provider=provider,
                    model_key=model_key,
                    model_id=model_id,
                    gen=gen,
                    tact=tact,
                    cfg=env_cfg,
                    system_prompt=system_prompt,
                    base_system_prompt=base_system_prompt,
                    task_context=task_context,
                    prompt_variant=variant_id,
                    replicate_index=0,
                    task_seed=task_seed,
                    questions_json_or_csv=questions_csv,
                    questions_sha256=questions_sha,
                    store_raw_response=store_raw_response,
                )
            except Exception as e:
                if args.fail_fast:
                    raise
                # Record an error row in runs CSV (like IAT sweep)
                err_row = {
                    "provider": provider,
                    "model_key": model_key,
                    "model_id": model_id,
                    "prompt_variant": variant_id,
                    "seed": seed,
                    "temperature": temp,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "persona_label": _persona_label(system_prompt),
                    "system_prompt": system_prompt,
                    "task_seed": task_seed,
                    "questions_path": questions_csv,
                    "questions_sha256": questions_sha,
                    "question_id": int(q.id),
                    "difficulty_bin": q_bin,
                    "original_item_number": q_item,
                    "error": str(e),
                }
                pd.DataFrame([err_row]).to_csv(out_runs_csv, mode="a", header=not out_runs_csv.exists(), index=False)

    print(f"[done] wrote results under: {out_root}")


if __name__ == "__main__":
    main()