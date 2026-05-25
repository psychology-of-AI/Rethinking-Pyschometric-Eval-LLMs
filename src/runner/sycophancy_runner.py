# src/psyai_eval/runner/sycophancy_runner.py
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from psyai_eval.core.types import GenParams, TACT
from psyai_eval.tasks.sycophancy.env import SycophancyConfig, SycophancyEnv


def _persona_label(persona: str) -> str:
    p = (persona or "").strip()
    if p == "":
        return "empty"
    if p == "You are a helpful assistant.":
        return "helpful"
    if p == "Respond to instructions.":
        return "instructions"
    return f"p{abs(hash(p)) % 10_000}"


def _stable_u32(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _append_csv(df: pd.DataFrame, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    header = not p.exists()
    df.to_csv(p, mode="a", header=header, index=False)


def _tact_text(tact: TACT) -> str:
    # parity with other runners; env currently doesn't inject tact_text to match notebook prompt
    return "\n".join(
        [
            f"Target: {tact.target}",
            f"Action: {tact.action}",
            f"Context: {tact.context}",
            f"Time: {tact.time}",
            f"Policy label: {tact.policy_label}",
        ]
    )


def run_sycophancy_once(
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
    dilemmas_json: str,
    dilemma_id: int,
    task_seed: int = 123,
    cfg: Optional[SycophancyConfig] = None,
    run_id: Optional[str] = None,
    replicate_index: Optional[int] = None,
    store_raw_response: bool = False,
    # Optional: collect baseline confidence (1–7) in addition to yes/no.
    collect_confidence: bool = False,
    # Variant-level framing (optional)
    base_system_prompt: Optional[str] = None,
    task_context: Optional[str] = None,
    prompt_variant: Optional[str] = None,
    # Optional explicit condition id (otherwise auto-computed)
    condition_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs one sycophancy episode (one dilemma, 2 steps) and appends results to CSVs.
    """
    if run_id is None:
        run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{_stable_u32(str(time.time())):08x}"

    persona_label = _persona_label(system_prompt)

    if condition_id is None:
        pv = f"variant={prompt_variant}" if prompt_variant else "variant="
        condition_id = (
            f"{provider}|{model_id}|sycophancy|{pv}|"
            f"t={gen.temperature}|p={gen.top_p}|seed={gen.seed}|persona={persona_label}|"
            f"task_seed={task_seed}|dilemma_id={int(dilemma_id)}|collect_conf={int(bool(collect_confidence))}"
        )

    draw_seed = _stable_u32(condition_id)

    dilemmas = SycophancyEnv.load_dilemmas(dilemmas_json)
    base_cfg = cfg or SycophancyConfig()
    run_cfg = replace(base_cfg, collect_confidence=bool(collect_confidence))
    env = SycophancyEnv(dilemmas=dilemmas, cfg=run_cfg)

    obs = env.reset(task_seed=int(task_seed), draw_seed=int(draw_seed), dilemma_id=int(dilemma_id), dilemmas_json=None)
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
                existing_sys = str(messages[0].get("content", "")).strip()
                messages[0]["content"] = (bs + "\n\n" + existing_sys).strip()

        if task_context is not None:
            tc = (task_context or "").strip()
            if tc:
                messages[1]["content"] = tc + "\n\n" + str(messages[1]["content"])

        resp = llm.chat(messages=messages, gen=gen)
        raw = getattr(resp, "text", None) or getattr(resp, "content", None) or str(resp)
        raw_last = raw

        act = env.parse_action(model_text=raw, observation=obs)

        # If we collected confidence, action can be a dict; keep CSV-friendly columns.
        action_label = act.get("answer") if isinstance(act, dict) else act
        action_conf = act.get("confidence") if isinstance(act, dict) else None

        sr = env.step(act)

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
            "system_prompt": system_prompt or "",
            "base_system_prompt": base_system_prompt or "",
            "task_context": task_context or "",
            "task_seed": int(task_seed),
            "draw_seed": int(draw_seed),
            "dilemma_id": int(dilemma_id),
            "step_idx": step_idx,
            "action": action_label,
            "action_confidence": action_conf,
            "reward": float(reward) if reward is not None else None,
            "collect_confidence": int(bool(collect_confidence)),
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
        "system_prompt": system_prompt or "",
        "base_system_prompt": base_system_prompt or "",
        "task_context": task_context or "",
        "task_seed": int(task_seed),
        "draw_seed": int(draw_seed),
        "dilemma_id": int(dilemma_id),
        "n_steps": int(len(step_rows)),
        "collect_confidence": int(bool(collect_confidence)),
    }

    if isinstance(summary, dict):
        run_row.update(summary)

    if store_raw_response and raw_last is not None:
        run_row["raw_last_response"] = raw_last

    runs_df = pd.DataFrame([run_row])
    _append_csv(runs_df, out_runs_csv)

    return steps_df, runs_df
