# -*- coding: utf-8 -*-
"""
Honesty runner: executes one 2-step honesty episode (one question) with an LLM.

Conventions (mirrors other runners in psyai_eval):
- condition_id is stable and includes provider/model/gen/persona/question_id/task_seed/(questions_sha)
- draw_seed is derived from condition_id for reproducibility across stochastic components
- writes two CSVs:
    * out_steps_csv: per-step rows
    * out_runs_csv: single-row run summary
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from psyai_eval.core.types import GenParams, TACT
from psyai_eval.tasks.base import StepResult
from psyai_eval.tasks.honesty.env import HonestyConfig, HonestyEnv, HonestyQuestion


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
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big", signed=False)


def _append_csv(df: pd.DataFrame, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def run_honesty_once(
    *,
    llm: Any,
    questions: List[HonestyQuestion],
    question_id: int,
    out_dir: Union[str, Path],
    out_steps_csv: Union[str, Path],
    out_runs_csv: Union[str, Path],
    provider: str,
    model_key: str,
    model_id: str,
    gen: GenParams,
    tact: Optional[TACT] = None,
    cfg: Optional[HonestyConfig] = None,
    # Prompt framing
    system_prompt: str = "",
    base_system_prompt: Optional[str] = None,
    task_context: Optional[str] = None,
    prompt_variant: str = "single",
    replicate_index: int = 0,
    task_seed: int = 123,
    # Dataset provenance
    questions_json_or_csv: Optional[str] = None,
    questions_sha256: Optional[str] = None,
    # Logging
    store_raw_response: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs one 2-step episode for a single question_id.

    Returns:
      (steps_df, runs_df) for convenience (they are also appended to CSVs).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve questions hash (if not provided)
    q_sha = (questions_sha256 or "").strip()
    if not q_sha and questions_json_or_csv:
        p = Path(questions_json_or_csv)
        if p.exists():
            q_sha = hashlib.sha256(p.read_bytes()).hexdigest()

    persona_label = _persona_label(system_prompt)

    env_cfg = cfg or HonestyConfig()
    env = HonestyEnv(questions=questions, cfg=env_cfg)
    obs = env.reset(task_seed=task_seed, draw_seed=0, question_id=question_id)

    # Stable condition id (include what changes the condition)
    condition_id = "|".join(
        [
            f"provider={provider}",
            f"model_key={model_key}",
            f"model_id={model_id}",
            f"variant={prompt_variant}",
            f"rep={int(replicate_index)}",
            f"seed={gen.seed}",
            f"temp={gen.temperature}",
            f"top_p={gen.top_p}",
            f"max_tokens={gen.max_tokens}",
            f"persona={persona_label}",
            f"task_seed={int(task_seed)}",
            f"question_id={int(question_id)}",
            f"questions_sha256={q_sha}",
        ]
    )

    run_id = hashlib.sha256(condition_id.encode("utf-8")).hexdigest()[:16]
    draw_seed = _stable_u32(condition_id)

    # Re-seed env with draw_seed for parity (env doesn't use it today)
    env.reset(task_seed=task_seed, draw_seed=draw_seed, question_id=question_id)

    tact_text = ""
    if tact is not None:
        tact_text = "\n".join(
            [
                f"Target: {tact.target}",
                f"Action: {tact.action}",
                f"Context: {tact.context}",
                f"Time: {tact.time}",
                f"Policy label: {tact.policy_label}",
            ]
        )

    steps: List[Dict[str, Any]] = []
    started_at = time.time()

    step_idx = 0
    done = False
    while not done and step_idx < 10:
        messages = env.render_prompt(observation=obs, system_prompt=system_prompt, tact_text=tact_text)

        # Preserve env's system constraints: prepend base_system_prompt into system message
        if base_system_prompt:
            messages[0]["content"] = (base_system_prompt.strip() + "\n\n" + (messages[0]["content"] or "")).strip()

        # Prepend task_context into user message
        if task_context:
            messages[1]["content"] = (task_context.strip() + "\n\n" + (messages[1]["content"] or "")).strip()

        resp = llm.chat(messages=messages, gen=gen)
        raw = getattr(resp, "text", None) or getattr(resp, "content", None) or str(resp)

        action = env.parse_action(raw, obs)
        sr: StepResult = env.step(action)

        # Base metadata
        step_row: Dict[str, Any] = {
            "run_id": run_id,
            "condition_id": condition_id,
            "provider": provider,
            "model_key": model_key,
            "model_id": model_id,
            "prompt_variant": prompt_variant or "",
            "replicate_index": int(replicate_index),
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
            "questions_path": str(questions_json_or_csv or ""),
            "questions_sha256": str(q_sha or ""),
            "question_id": int(question_id),
            "step": int(step_idx),
            "reward": float(sr.reward) if sr.reward is not None else None,
        }

        if isinstance(sr.info, dict):
            for k2, v2 in sr.info.items():
                if k2 not in step_row:
                    step_row[k2] = v2

        if store_raw_response:
            step_row["raw_response"] = raw

        steps.append(step_row)

        obs = sr.observation
        done = bool(sr.done)
        step_idx += 1

    steps_df = pd.DataFrame(steps)
    _append_csv(steps_df, out_steps_csv)

    summary = env.summarize_run()
    # reward is float(y) when scored; or None
    reward = None
    if isinstance(summary, dict):
        yb = summary.get("is_correct_em", None)
        if yb is True:
            reward = 1.0
        elif yb is False:
            reward = 0.0

    run_row: Dict[str, Any] = {
        "run_id": run_id,
        "condition_id": condition_id,
        "provider": provider,
        "model_key": model_key,
        "model_id": model_id,
        "prompt_variant": prompt_variant or "",
        "replicate_index": int(replicate_index),
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
        "questions_path": str(questions_json_or_csv or ""),
        "questions_sha256": str(q_sha or ""),
        "question_id": int(question_id),
        "n_steps": int(step_idx),
        "runtime_s": float(time.time() - started_at),
        "reward": reward,
    }

    if isinstance(summary, dict):
        for k, v in summary.items():
            if k not in run_row:
                run_row[k] = v

    runs_df = pd.DataFrame([run_row])
    _append_csv(runs_df, out_runs_csv)

    return steps_df, runs_df
