# scripts/sweep_tpb_variants.py
from __future__ import annotations

import hashlib
import argparse
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from psyai_eval.core.types import GenParams, TACT
from psyai_eval.llms.openai_compat import OpenAICompatChatClient
from psyai_eval.runner.tpb_likert_runner import run_tpb_likert_once


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_env_key(*names: str) -> Optional[str]:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


def _get_openrouter_key() -> str:
    key = _get_env_key("OPENROUTER_API_KEY")
    if key:
        return key
    raise RuntimeError("Missing OpenRouter API key. Set env var OPENROUTER_API_KEY.")


def _models_from_config(cfg: dict, provider: str, model_keys: Optional[List[str]]) -> List[Tuple[str, str]]:
    block = cfg[provider]
    if model_keys:
        return [(k, str(block[k])) for k in model_keys]
    return [(k, str(v)) for k, v in block.items()]

def _resolve_path(p: str, *, config_path: str) -> Path:
    """
    Resolve a possibly-relative path:
      1) as given
      2) relative to config file directory
      3) relative to current working directory
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
    raise FileNotFoundError(f"Could not resolve path: {p} (config={config_path})")

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
    Load personas from configs/selected_personas.json produced by PersonaHub selector.
    Expected structure:
      { "selections": { "seed_42": { "persona_prompts": [...] }, ... } }
    """
    p = _resolve_path(selected_path, config_path=config_path)
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
        prompts = prompts[: int(limit)]

    out = []
    if prepend:
        out.extend(prepend)
    out.extend(prompts)
    return out


def _persona_label(persona: str) -> str:
    p = (persona or "").strip()
    if p == "":
        return "empty"
    if p == "You are a helpful assistant.":
        return "helpful"
    if p == "Respond to instructions.":
        return "instructions"
    digest = hashlib.sha256(p.encode("utf-8")).hexdigest()
    return f"p{digest[:4]}"


def _already_done_keys(runs_csv: Path) -> set:
    if not runs_csv.exists():
        return set()
    try:
        df = pd.read_csv(runs_csv, on_bad_lines="skip", engine="python")
    except Exception:
        return set()

    # TPB runner writes variant id as gen_extra__prompt_variant, not prompt_variant
    if "gen_extra__prompt_variant" in df.columns:
        df["prompt_variant"] = df["gen_extra__prompt_variant"]

    needed = {"prompt_variant", "model_key", "seed", "temperature", "top_p", "persona_label", "system_prompt"}
    if not needed.issubset(df.columns):
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
            ))
        except (ValueError, TypeError):
            n_skipped += 1
    if n_skipped:
        print(f"[resume] WARNING: skipped {n_skipped} corrupt rows in {runs_csv}")
    return keys


def _make_openrouter_client(model: str, base_url: Optional[str], headers: dict) -> OpenAICompatChatClient:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON config with grid + variants")
    ap.add_argument("--out_root", default="results/tpb_sweeps", help="Root output directory")
    ap.add_argument("--skip_variant_path", action="store_true", help="If used, skips putting outputs in the /exp_name/variant_id subpath")

    ap.add_argument("--base_url", default=None)
    ap.add_argument("--max_tokens", type=int, default=700)
    ap.add_argument("--max_attempts", type=int, default=5)
    ap.add_argument("--resume", action="store_true", help="Skip runs already present in tpb_likert_runs.csv")
    ap.add_argument("--fail_fast", action="store_true", help="Stop on first error (default: continue)")
    args = ap.parse_args()

    cfg = _load_json(args.config)

    provider = cfg.get("provider", "openrouter")
    if provider != "openrouter":
        raise ValueError("This sweep script currently supports provider=openrouter only.")

    models_cfg = _load_json(cfg["models_config"])
    model_pairs = _models_from_config(models_cfg, provider, cfg.get("model_keys"))

    seeds = cfg["grid"]["seeds"]
    temps = cfg["grid"]["temperatures"]
    top_ps = cfg["grid"]["top_ps"]
    
    grid = cfg.get("grid", {})
    persona_mode = str(grid.get("persona_prompts_mode", "direct")).strip().lower()
    if persona_mode not in ("direct", "selected_personas"):
        raise ValueError("grid.persona_prompts_mode must be 'direct' or 'selected_personas'")

    if persona_mode == "selected_personas":
        sel = grid.get("persona_prompts_selected", {}) or {}
        selected_path = str(sel.get("path", "configs/selected_personas.json"))
        selected_seed = int(sel.get("seed"))
        selected_field = str(sel.get("field", "persona_prompts"))
        selected_limit = sel.get("limit", None)
        selected_prepend = sel.get("prepend", None)
        persona_prompts = _load_persona_prompts_from_selected(
            selected_path,
            config_path=args.config,
            seed=selected_seed,
            field=selected_field,
            limit=selected_limit,
            prepend=selected_prepend,
        )
    else:
        persona_prompts = grid["persona_prompts"]

    headers = {
        "HTTP-Referer": (cfg.get("openrouter_referer", os.environ.get("OPENROUTER_REFERER", "")) or "").strip(),
        "X-Title": (cfg.get("openrouter_title", os.environ.get("OPENROUTER_TITLE", "psyai_eval")) or "").strip(),
    }

    # single client; we swap model each run
    llm = _make_openrouter_client(model_pairs[0][1], args.base_url, headers)

    exp_name = cfg.get("experiment_name", "tpb_prompt_sweep")
    run_batch_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

    variants = [v for v in cfg["variants"] if v.get("enabled", True)]
    for v in variants:
        variant_id = v["variant_id"]
        out_dir = os.path.join(args.out_root, exp_name, variant_id)
        if args.skip_variant_path:
            out_dir = Path(args.out_root)
        _ensure_dir(out_dir)

        out_csv = os.path.join(out_dir, "tpb_likert.csv")
        out_runs_csv = os.path.join(out_dir, "tpb_likert_runs.csv")

        tact = TACT(**v["tact"])
        base_system_prompt = v.get("base_system_prompt", None)
        task_context = v.get("task_context", None)
        likert_items = v.get("likert_items", cfg.get("likert_items", None))

        # Variant-level persona_prompts override grid-level (used by steering configs).
        # Falls back to grid persona_prompts for baseline/neutral configs.
        effective_persona_prompts = list(v.get("persona_prompts") or persona_prompts)

        done = _already_done_keys(Path(out_runs_csv)) if args.resume else set()

        total = len(model_pairs) * len(seeds) * len(temps) * len(top_ps) * len(effective_persona_prompts)
        bar = tqdm(total=total, desc=f"TPB sweep [{variant_id}]", unit="run", ncols=110)

        for model_key, model_id in model_pairs:
            llm.model = model_id

            for seed in seeds:
                for temp in temps:
                    for top_p in top_ps:
                        for persona in effective_persona_prompts:
                            persona = persona or ""
                            persona_lab = _persona_label(persona)

                            resume_key = (variant_id, model_key, int(seed), float(temp), float(top_p), persona_lab, persona)
                            if resume_key in done:
                                bar.update(1)
                                continue

                            bar.set_postfix(
                                {
                                    "model": model_key,
                                    "seed": seed,
                                    "t": temp,
                                    "p": top_p,
                                    "persona": persona_lab,
                                }
                            )

                            gen = GenParams(
                                temperature=float(temp),
                                top_p=float(top_p),
                                max_tokens=int(cfg.get("max_tokens", args.max_tokens)),
                                seed=int(seed),
                                extra={"prompt_variant": variant_id},
                            )

                            run_id = f"{run_batch_id}_{variant_id}_{model_key}_seed{seed}_t{temp}_p{top_p}_{persona_lab}"

                            try:
                                run_tpb_likert_once(
                                    llm=llm,
                                    tact=tact,
                                    gen=gen,
                                    system_prompt=persona,
                                    base_system_prompt=base_system_prompt,
                                    task_context=task_context,
                                    items_override=likert_items,
                                    out_csv=out_csv,
                                    out_runs_csv=out_runs_csv,
                                    run_id=run_id,
                                    store_raw_response=bool(cfg.get("store_raw_response", False)),
                                    model_key=model_key,
                                    model_id=model_id,
                                    max_attempts=int(cfg.get("max_attempts", args.max_attempts)),
                                    continue_on_error=True,
                                    steering=v.get("steering"),
                                )
                            except Exception as e:
                                if args.fail_fast:
                                    raise
                                # Record the failure minimally so the sweep can continue.
                                err_row = pd.DataFrame(
                                    [
                                        {
                                            "prompt_variant": variant_id,
                                            "model_key": model_key,
                                            "model_id": model_id,
                                            "seed": int(seed),
                                            "temperature": float(temp),
                                            "top_p": float(top_p),
                                            "persona_label": persona_lab,
                                            "system_prompt": persona,
                                            "error": repr(e),
                                        }
                                    ]
                                )
                                header = not Path(out_runs_csv).exists()
                                err_row.to_csv(out_runs_csv, mode="a", header=header, index=False)
                            finally:
                                bar.update(1)

        bar.close()

    print(f"Done. Outputs under: {os.path.join(args.out_root, exp_name)}")


if __name__ == "__main__":
    main()