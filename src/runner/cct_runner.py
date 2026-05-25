# -*- coding: utf-8 -*-
"""
CCT runner: executes one full Columbia Card Task (cold) run with an LLM.

Key design points:
- Task randomness is controlled by (task_seed, draw_seed). draw_seed is derived from run metadata so
  you get stochastic decks across replicates while remaining reproducible.
- Prompt framing supports:
  * base_system_prompt: "participant in a study" instruction (variant-level)
  * system_prompt: persona/system perturbation (grid-level)
  * task_context: extra task description (variant-level), prepended to the user message
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from psyai_eval.core.types import GenParams, TACT
from psyai_eval.perturbations.steering_utils import SteeringDict, apply_steering, steering_meta, steering_to_json
from psyai_eval.tasks.cct.env import CCTConfig, CCTEnv
from psyai_eval.tasks.base import StepResult


def _persona_label(persona: str) -> str:
    p = (persona or "").strip()
    if p == "":
        return "empty"
    if p == "You are a helpful assistant.":
        return "helpful"
    if p == "Respond to instructions.":
        return "instructions"
    # Use SHA-256 for a label that is stable across Python processes.
    # (Python's built-in hash() is randomised per-process via PYTHONHASHSEED,
    # so hash()-based labels differ between the original run and --resume,
    # causing the resume key to never match and re-running all completed work.)
    digest = hashlib.sha256(p.encode("utf-8")).hexdigest()
    return f"p{digest[:4]}"


def _stable_u32(s: str) -> int:
    """Deterministic 32-bit int from a string."""
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _sanitize_for_csv(s: str) -> str:
    """
    Replace literal newlines with the two-char escape sequence \\n before writing to CSV.

    pandas' C parser reads line-by-line before interpreting RFC 4180 quoting,
    so embedded newlines in quoted fields split a single logical row across
    multiple physical lines, producing wrong field counts on read-back
    ("Expected N fields, saw M").  Escaping at the write site prevents this
    entirely; readers round-trip back with .replace('\\\\n', '\\n').
    """
    return s.replace("\n", "\\n") if s else s


def _append_csv(df: pd.DataFrame, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    header = not p.exists()
    df.to_csv(p, mode="a", header=header, index=False)


def _tact_text(tact: TACT) -> str:
    return "\n".join(
        [
            f"Target: {tact.target}",
            f"Action: {tact.action}",
            f"Context: {tact.context}",
            f"Time: {tact.time}",
            f"Policy label: {tact.policy_label}",
        ]
    )


def run_cct_once(
    *,
    llm: Any,
    tact: TACT,
    gen: GenParams,
    system_prompt: str,
    out_steps_csv: str,
    out_runs_csv: str,
    model_key: str,
    model_id: str,
    provider: str,
    task_seed: int = 123,
    cfg: Optional[CCTConfig] = None,
    run_id: Optional[str] = None,
    replicate_index: Optional[int] = None,
    store_raw_response: bool = False,
    # Variant-level framing (optional)
    base_system_prompt: Optional[str] = None,
    task_context: Optional[str] = None,
    prompt_variant: Optional[str] = None,
    # Optional explicit condition id (otherwise auto-computed)
    condition_id: Optional[str] = None,
    # RQ2: optional steering intervention
    steering: SteeringDict = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs one CCT episode and appends results to CSVs.

    Returns:
      steps_df: per-step dataframe for this run
      runs_df:  single-row dataframe summarizing this run
    """
    if run_id is None:
        # short, file-friendly
        run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{_stable_u32(str(time.time())):08x}"

    persona_label = _persona_label(system_prompt)

    if condition_id is None:
        pv = f"variant={prompt_variant}" if prompt_variant else "variant="
        condition_id = (
            f"{provider}|{model_id}|cct|{pv}|"
            f"t={gen.temperature}|p={gen.top_p}|seed={gen.seed}|persona={persona_label}|"
            f"task_seed={task_seed}"
        )

    # draw_seed varies across conditions but is reproducible
    draw_seed = _stable_u32(condition_id)

    env = CCTEnv(cfg=cfg or CCTConfig())
    obs = env.reset(task_seed=int(task_seed), draw_seed=int(draw_seed))

    tact_txt = _tact_text(tact)

    step_rows: List[Dict[str, Any]] = []
    raw_last: Optional[str] = None

    done = False
    step_idx = 0

    while not done:
        messages = env.render_prompt(observation=obs, system_prompt=system_prompt, tact_text=tact_txt)

        # Inject variant-level framing (no env changes needed)
        if base_system_prompt is not None:
            bs = (base_system_prompt or "").strip()
            if bs:
                # IMPORTANT: preserve env-provided system content (incl. JSON-only instruction)
                existing_sys = str(messages[0].get("content", "")).strip()
                messages[0]["content"] = (bs + "\n\n" + existing_sys).strip()


        if task_context is not None:
            tc = (task_context or "").strip()
            if tc:
                messages[1]["content"] = tc + "\n\n" + str(messages[1]["content"])

        # RQ2: inject steering on top of all other framing. No-op if steering is None.
        apply_steering(messages, steering)

        resp = llm.chat(messages=messages, gen=gen)
        raw = getattr(resp, "text", None) or getattr(resp, "content", None) or str(resp)
        raw_last = raw

        # Parse an integer k. Be forgiving: first integer token.
        # Task-owned parsing (JSON-first + fallback + clamp)
        k = env.parse_action(model_text=raw, observation=obs)

        sr = env.step(k)
        obs = sr.observation
        reward = sr.reward
        done = sr.done
        info = sr.info

        row: Dict[str, Any] = {
            "run_id": run_id,
            "condition_id": condition_id,
            "provider": provider,
            "model_key": model_key,
            "model_id": model_id,
            "prompt_variant": prompt_variant or "",
            "replicate_index": replicate_index,
            "seed": gen.seed,
            "temperature": gen.temperature,
            "top_p": gen.top_p,
            "max_tokens": gen.max_tokens,
            "persona_label": persona_label,
            "system_prompt": _sanitize_for_csv(system_prompt or ""),
            "base_system_prompt": _sanitize_for_csv(base_system_prompt or ""),
            "task_context": _sanitize_for_csv(task_context or ""),
            "task_seed": int(task_seed),
            "draw_seed": int(draw_seed),
            "step_idx": step_idx,
            "action_k": int(k),
            "reward": float(reward),
        }

        if isinstance(info, dict):
            for k2, v2 in info.items():
                if k2 not in row:
                    row[k2] = v2

        if store_raw_response:
            row["raw_response"] = raw

        step_rows.append(row)
        step_idx += 1

    steps_df = pd.DataFrame(step_rows)
    _append_csv(steps_df, out_steps_csv)

    summary = env.summarize_run()
    run_row: Dict[str, Any] = {
        "run_id": run_id,
        "condition_id": condition_id,
        "provider": provider,
        "model_key": model_key,
        "model_id": model_id,
        "prompt_variant": prompt_variant or "",
        "replicate_index": replicate_index,
        "seed": gen.seed,
        "temperature": gen.temperature,
        "top_p": gen.top_p,
        "max_tokens": gen.max_tokens,
        "persona_label": persona_label,
        "system_prompt": _sanitize_for_csv(system_prompt or ""),
        "base_system_prompt": _sanitize_for_csv(base_system_prompt or ""),
        "task_context": _sanitize_for_csv(task_context or ""),
        "task_seed": int(task_seed),
        "draw_seed": int(draw_seed),
        "n_steps": int(len(step_rows)),
        "steering_json": steering_to_json(steering),
        **steering_meta(steering),
    }

    if isinstance(summary, dict):
        run_row.update(summary)

    if store_raw_response and raw_last is not None:
        run_row["raw_last_response"] = raw_last

    runs_df = pd.DataFrame([run_row])
    _append_csv(runs_df, out_runs_csv)

    return steps_df, runs_df