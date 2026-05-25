# scripts/config_based_sweeps/sweep_sycophancy_variants.py
from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm.auto import tqdm

from psyai_eval.core.types import GenParams, TACT
from psyai_eval.llms.openai_compat import OpenAICompatChatClient
from psyai_eval.runner.sycophancy_runner import run_sycophancy_once
from psyai_eval.tasks.sycophancy.env import SycophancyEnv


class _DummyLLM:
    """Simple mock LLM for sweeps."""

    def __init__(self, mode: str = "agree_user"):
        self.mode = mode

    def chat(self, *, messages: List[Dict[str, str]], gen: GenParams):
        # Baseline: answer "no" (and optionally include a confidence)
        # After suggestion: either agree with user ("yes"/"no") or keep baseline.
        user = (messages[-1].get("content") or "").lower()

        if "i would say" in user:
            if self.mode == "agree_user":
                if "i would say yes" in user:
                    return type("Resp", (), {"text": "yes"})()
                if "i would say no" in user:
                    return type("Resp", (), {"text": "no"})()
                return type("Resp", (), {"text": "yes"})()
            elif self.mode == "keep_baseline":
                return type("Resp", (), {"text": "no"})()

        # baseline (confidence-friendly)
        if "json" in user and "confidence" in user:
            return type("Resp", (), {"text": '{"answer":"no","confidence":6}'})()
        return type("Resp", (), {"text": "no"})()


def _get_openrouter_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    return key


def _load_model_id(models_config_path: str, provider: str, model_key: str) -> str:
    with open(models_config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if provider not in data:
        raise KeyError(f"Provider '{provider}' not found in {models_config_path}")
    if model_key not in data[provider]:
        raise KeyError(
            f"Model key '{model_key}' not found under provider '{provider}' in {models_config_path}"
        )
    return str(data[provider][model_key])


def _persona_label(persona: str) -> str:
    p = (persona or "").strip()
    if p == "":
        return "empty"
    if p == "You are a helpful assistant.":
        return "helpful"
    if p == "Respond to instructions.":
        return "instructions"
    return f"p{abs(hash(p)) % 10_000}"


def _existing_keys(runs_csv: Path) -> Set[Tuple[Any, ...]]:
    """Resume keys for a variant's runs CSV. Skips corrupt rows rather than raising."""
    if not runs_csv.exists():
        return set()
    try:
        df = pd.read_csv(runs_csv, on_bad_lines="skip")
    except Exception as e:
        print(f"[resume] WARNING: could not read {runs_csv}: {e} — treating as empty")
        return set()

    needed = ["provider", "model_id", "prompt_variant", "temperature", "top_p", "seed", "persona_label", "task_seed", "dilemma_id"]
    if any(c not in df.columns for c in needed):
        missing = [c for c in needed if c not in df.columns]
        print(f"[resume] WARNING: {runs_csv} missing columns {missing} — treating as empty")
        return set()

    keys: Set[Tuple[Any, ...]] = set()
    n_skipped = 0
    for row in df.itertuples(index=False):
        try:
            cc = int(float(row.collect_confidence)) if hasattr(row, "collect_confidence") and str(getattr(row, "collect_confidence", "")) not in ("", "nan") else 0
            keys.add((
                str(row.provider),
                str(row.model_id),
                str(row.prompt_variant),
                float(row.temperature),
                float(row.top_p),
                int(float(row.seed)),
                str(row.persona_label),
                int(float(row.task_seed)),
                int(float(row.dilemma_id)),
                int(cc),
            ))
        except (ValueError, TypeError):
            n_skipped += 1
    if n_skipped:
        print(f"[resume] WARNING: skipped {n_skipped} corrupt rows in {runs_csv}")
    return keys


@dataclass(frozen=True)
class _PlanItem:
    variant_id: str
    prompt_variant: str
    base_system_prompt: Optional[str]
    task_context: Optional[str]
    tact: TACT

    provider: str
    model_key: str
    model_id: str

    seed: int
    temperature: float
    top_p: float
    persona_prompt: str
    persona_label: str

    task_seed: int
    dilemma_id: int

    collect_confidence: bool

    out_steps_csv: str
    out_runs_csv: str

    max_tokens: int
    store_raw_response: bool
    dilemmas_json: str


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


def main() -> None:
    p = argparse.ArgumentParser(description="Config-based sweeps for sycophancy task.")
    p.add_argument("--config", required=True, help="Path to sycophancy sweep config JSON.")
    p.add_argument("--out_root", required=True, help="Output root directory (e.g., results/sycophancy_sweeps).")
    p.add_argument("--resume", action="store_true", help="Skip runs already present in sycophancy_runs.csv.")
    p.add_argument("--provider_override", default=None, help="Override provider in config (e.g., mock).")
    p.add_argument("--mock_mode", choices=["agree_user", "keep_baseline"], default="agree_user")
    p.add_argument("--max_runs", type=int, default=None, help="Optional cap on number of runs (for smoke tests).")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    experiment_name = str(cfg["experiment_name"])
    provider = str(args.provider_override or cfg.get("provider", "openrouter"))
    models_config = str(cfg.get("models_config", "configs/openrouter_models.json"))
    model_keys: List[str] = list(cfg.get("model_keys", []))
    grid: Dict[str, Any] = dict(cfg.get("grid", {}))
    variants: List[Dict[str, Any]] = list(cfg.get("variants", []))

    max_tokens = int(cfg.get("max_tokens", 64))
    store_raw = bool(cfg.get("store_raw_response", False))

    syc = dict(cfg.get("sycophancy", {}))
    dilemmas_json = str(syc.get("dilemmas_json", "configs/sycophancy/sycophancy_dilemmas.json"))
    dilemma_ids = syc.get("dilemma_ids", "ALL")
    collect_confidence = bool(syc.get("collect_confidence", False))

    if dilemma_ids == "ALL" or dilemma_ids is None:
        dilemmas = SycophancyEnv.load_dilemmas(dilemmas_json)
        dilemma_id_list = [int(d["id"]) for d in dilemmas]
    else:
        dilemma_id_list = [int(x) for x in dilemma_ids]

    seeds = [int(x) for x in grid.get("seeds", [42])]
    temperatures = [float(x) for x in grid.get("temperatures", [0.7])]
    top_ps = [float(x) for x in grid.get("top_ps", [1.0])]
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
        persona_prompts = [str(x) for x in grid.get("persona_prompts", [""])]
    task_seed = int(grid.get("task_seed", 123))

    out_root = Path(args.out_root) / experiment_name
    out_root.mkdir(parents=True, exist_ok=True)

    # Copy config for provenance
    try:
        shutil.copy2(args.config, out_root / Path(args.config).name)
    except Exception:
        pass

    # Build a plan so tqdm can show a true total.
    plan: List[_PlanItem] = []

    for variant in variants:
        variant_id = str(variant.get("variant_id", "variant"))
        base_system_prompt = variant.get("base_system_prompt", None)
        task_context = variant.get("task_context", None)
        tact_dict = dict(variant.get("tact", {}))
        prompt_variant = variant_id

        tact = TACT(**tact_dict) if tact_dict else TACT(
            target="I",
            action="answer yes/no",
            context="given a moral dilemma",
            time="right now",
            policy_label=variant_id,
        )

        variant_dir = out_root / variant_id
        variant_dir.mkdir(parents=True, exist_ok=True)
        out_steps_csv = variant_dir / "sycophancy_steps.csv"
        out_runs_csv = variant_dir / "sycophancy_runs.csv"

        if args.resume:
            try:
                seen = _existing_keys(out_runs_csv)
                print(f"[resume] variant={variant_id} — found {len(seen)} completed keys in {out_runs_csv}")
            except Exception as e:
                print(f"[resume] WARNING: failed to load done_keys for variant={variant_id} ({e}) — will re-run all")
                seen = set()
        else:
            seen = set()

        model_id_by_key: Dict[str, str] = {}
        for mk in model_keys:
            if provider == "mock":
                model_id_by_key[mk] = f"mock/{args.mock_mode}"
            else:
                model_id_by_key[mk] = _load_model_id(models_config, provider, mk)

        for mk in model_keys:
            mid = model_id_by_key[mk]
            for seed in seeds:
                for temp in temperatures:
                    for top_p in top_ps:
                        for persona in persona_prompts:
                            plabel = _persona_label(persona)
                            for did in dilemma_id_list:
                                key = (
                                    provider,
                                    str(mid),
                                    str(prompt_variant),
                                    float(temp),
                                    float(top_p),
                                    int(seed),
                                    str(plabel),
                                    int(task_seed),
                                    int(did),
                                    int(bool(collect_confidence)),
                                )
                                if key in seen:
                                    continue

                                plan.append(
                                    _PlanItem(
                                        variant_id=variant_id,
                                        prompt_variant=prompt_variant,
                                        base_system_prompt=base_system_prompt,
                                        task_context=task_context,
                                        tact=tact,
                                        provider=provider,
                                        model_key=str(mk),
                                        model_id=str(mid),
                                        seed=int(seed),
                                        temperature=float(temp),
                                        top_p=float(top_p),
                                        persona_prompt=str(persona),
                                        persona_label=str(plabel),
                                        task_seed=int(task_seed),
                                        dilemma_id=int(did),
                                        collect_confidence=bool(collect_confidence),
                                        out_steps_csv=str(out_steps_csv),
                                        out_runs_csv=str(out_runs_csv),
                                        max_tokens=int(max_tokens),
                                        store_raw_response=bool(store_raw),
                                        dilemmas_json=str(dilemmas_json),
                                    )
                                )

    if args.max_runs is not None:
        plan = plan[: int(args.max_runs)]

    if not plan:
        print("Nothing to run (everything already present, or empty plan).")
        return

    llm_cache: Dict[str, Any] = {}

    pbar = tqdm(plan, desc="Sycophancy", unit="run", dynamic_ncols=True)
    for item in pbar:
        pbar.set_postfix(
            {
                "variant": item.variant_id,
                "model": item.model_key,
                "temp": item.temperature,
                "seed": item.seed,
                "persona": item.persona_label,
                "dilemma": item.dilemma_id,
                "conf": int(bool(item.collect_confidence)),
            }
        )

        gen = GenParams(
            temperature=float(item.temperature),
            top_p=float(item.top_p),
            max_tokens=int(item.max_tokens),
            seed=int(item.seed),
        )

        if item.provider == "mock":
            cache_key = f"mock::{args.mock_mode}"
            if cache_key not in llm_cache:
                llm_cache[cache_key] = _DummyLLM(mode=args.mock_mode)
            llm = llm_cache[cache_key]
        else:
            if item.model_id not in llm_cache:
                llm_cache[item.model_id] = OpenAICompatChatClient(
                    api_key=_get_openrouter_key(),
                    base_url="https://openrouter.ai/api/v1",
                    model=str(item.model_id),
                    default_headers={
                        "HTTP-Referer": "http://localhost",
                        "X-Title": "psyai-eval-sycophancy",
                    },
                )
            llm = llm_cache[item.model_id]

        run_sycophancy_once(
            llm=llm,
            tact=item.tact,
            gen=gen,
            system_prompt=str(item.persona_prompt),
            base_system_prompt=item.base_system_prompt,
            task_context=item.task_context,
            prompt_variant=item.prompt_variant,
            out_steps_csv=str(item.out_steps_csv),
            out_runs_csv=str(item.out_runs_csv),
            model_key=str(item.model_key),
            model_id=str(item.model_id),
            provider=str(item.provider),
            dilemmas_json=str(item.dilemmas_json),
            dilemma_id=int(item.dilemma_id),
            task_seed=int(item.task_seed),
            store_raw_response=bool(item.store_raw_response),
            collect_confidence=bool(item.collect_confidence),
        )

    print(f"Done. Wrote results under: {out_root}")


if __name__ == "__main__":
    main()