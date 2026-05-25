# -*- coding: utf-8 -*-
"""
IAT runner: executes one full IAT episode (one prompt -> one response -> one bias score).

Design matches CCT runner:
- condition_id captures provider/model/prompt metadata
- draw_seed derived from condition_id for reproducible "luck" (passed to env for parity/future use)
- prompt framing supports:
  * base_system_prompt: variant-level system framing (prepended to env system message)
  * system_prompt: persona/system perturbation (passed into env)
  * task_context: variant-level extra context, prepended to user message
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from psyai_eval.core.types import GenParams, TACT
from psyai_eval.llms.openai_compat import OpenAICompatChatClient
from psyai_eval.tasks.iat.env import IATConfig, IATEnv


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
    return "\n".join(
        [
            f"Target: {tact.target}",
            f"Action: {tact.action}",
            f"Context: {tact.context}",
            f"Time: {tact.time}",
            f"Policy label: {tact.policy_label}",
        ]
    )


def _file_sha256(path: str) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_iat_once(
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
    stimuli_json: str,
    test_id: str,
    order_id: int,
    task_seed: int = 123,
    cfg: Optional[IATConfig] = None,
    run_id: Optional[str] = None,
    replicate_index: Optional[int] = None,
    store_raw_response: bool = False,
    # Variant-level framing (optional)
    base_system_prompt: Optional[str] = None,
    task_context: Optional[str] = None,
    prompt_variant: Optional[str] = None,
    # Optional explicit condition id (otherwise auto-computed)
    condition_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if run_id is None:
        run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{_stable_u32(str(time.time())):08x}"

    persona_label = _persona_label(system_prompt)
    stimuli_sha256 = _file_sha256(stimuli_json)

    if condition_id is None:
        pv = f"variant={prompt_variant}" if prompt_variant else "variant="
        condition_id = (
            f"{provider}|{model_id}|iat|{pv}|"
            f"test={test_id}|order={int(order_id)}|"
            f"t={gen.temperature}|p={gen.top_p}|seed={gen.seed}|persona={persona_label}|"
            f"task_seed={task_seed}|stimuli={stimuli_sha256[:8]}"
        )

    # draw_seed varies across conditions but is reproducible
    draw_seed = _stable_u32(condition_id)

    stimuli = IATEnv.load_stimuli(stimuli_json)
    env = IATEnv(stimuli=stimuli, cfg=cfg or IATConfig())

    obs = env.reset(
        task_seed=int(task_seed),
        draw_seed=int(draw_seed),
        test_id=str(test_id),
        order_id=int(order_id),
    )

    tact_txt = _tact_text(tact)
    messages = env.render_prompt(observation=obs, system_prompt=system_prompt, tact_text=tact_txt)

    # IMPORTANT: preserve env system message (includes any env-level instructions)
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

    action = env.parse_action(raw, obs)
    sr = env.step(action)

    # One-step task: log a single step row
    step_row: Dict[str, Any] = {
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
        "stimuli_json": str(stimuli_json),
        "stimuli_sha256": stimuli_sha256,
        "test_id": str(test_id),
        "order_id": int(order_id),
        "reward": float(sr.reward) if sr.reward is not None else None,
    }

    if isinstance(sr.info, dict):
        for k2, v2 in sr.info.items():
            if k2 not in step_row:
                step_row[k2] = v2

    if store_raw_response:
        step_row["raw_response"] = raw

    steps_df = pd.DataFrame([step_row])
    _append_csv(steps_df, out_steps_csv)

    # Run summary (single row)
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
        "stimuli_json": str(stimuli_json),
        "stimuli_sha256": stimuli_sha256,
        "test_id": str(test_id),
        "order_id": int(order_id),
        "n_steps": 1,
    }
    if isinstance(summary, dict):
        run_row.update(summary)

    if store_raw_response:
        run_row["raw_last_response"] = raw

    runs_df = pd.DataFrame([run_row])
    _append_csv(runs_df, out_runs_csv)

    return steps_df, runs_df


# ---------------------------
# CLI smoke tests
# ---------------------------

class _DummyResp:
    def __init__(self, text: str):
        self.text = text


class _DummyLLM:
    """Mock LLM that assigns every word to sa/sb or random (useful for testing without API)."""

    def __init__(self, mode: str = "random"):
        self.mode = mode

    def chat(self, *, messages: List[Dict[str, str]], gen: GenParams) -> _DummyResp:
        user = messages[-1]["content"]

        # Extract labels from prompt
        import re
        m = re.search(r'assign one of "([^"]+)" or "([^"]+)"', user, re.IGNORECASE)
        sa = m.group(1) if m else "GroupA"
        sb = m.group(2) if m else "GroupB"

        # Extract attribute list (comma-separated)
        m2 = re.search(r"The words are:\s*(.*)", user)
        words: List[str] = []
        if m2:
            wline = m2.group(1).splitlines()[0]
            words = [w.strip() for w in wline.split(",") if w.strip()]

        import random
        rng = random.Random(_stable_u32(f"iat-dummy|{gen.seed}|{self.mode}"))
        out = []
        for w in words:
            if self.mode == "all_sa":
                out.append(f"{w} - {sa}")
            elif self.mode == "all_sb":
                out.append(f"{w} - {sb}")
            else:
                out.append(f"{w} - {rng.choice([sa, sb])}")
        return _DummyResp("\n".join(out))


def _get_openrouter_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    return key


def main() -> None:
    p = argparse.ArgumentParser(description="Run a single IAT episode (runner-level).")
    p.add_argument("--stimuli_json", required=True)
    p.add_argument("--test_id", required=True)
    p.add_argument("--order_id", type=int, default=0)
    p.add_argument("--task_seed", type=int, default=123)

    p.add_argument("--out_dir", default="results/iat_single_runner")
    p.add_argument("--store_raw_response", action="store_true")

    # Prompt framing
    p.add_argument("--system_prompt", default="")
    p.add_argument("--base_system_prompt", default=None)
    p.add_argument("--task_context", default=None)
    p.add_argument("--prompt_variant", default="single")

    # LLM settings
    p.add_argument("--provider", default="openrouter", choices=["openrouter", "mock"])
    p.add_argument("--model_key", default="gpt4o_mini")
    p.add_argument("--model_id", default="openai/gpt-4o-mini")  # OpenRouter model id
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_tokens", type=int, default=900)
    p.add_argument("--seed", type=int, default=42)

    # Mock mode
    p.add_argument("--mock_mode", choices=["random", "all_sa", "all_sb"], default="random")

    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_steps = out_dir / "iat_steps.csv"
    out_runs = out_dir / "iat_runs.csv"

    gen = GenParams(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
        seed=int(args.seed),
    )

    tact = TACT(
        target="I",
        action="categorize each word into one of two labels",
        context="quickly and intuitively",
        time="right now",
        policy_label="Neutral",
    )

    if args.provider == "mock":
        llm = _DummyLLM(mode=args.mock_mode)
        provider = "mock"
    else:
        llm = OpenAICompatChatClient(
            api_key=_get_openrouter_key(),
            base_url="https://openrouter.ai/api/v1",
            model=str(args.model_id),
            default_headers={"HTTP-Referer": "http://localhost", "X-Title": "psyai-eval-iat"},
        )
        provider = "openrouter"

    run_iat_once(
        llm=llm,
        tact=tact,
        gen=gen,
        system_prompt=str(args.system_prompt),
        base_system_prompt=args.base_system_prompt,
        task_context=args.task_context,
        prompt_variant=str(args.prompt_variant),
        out_steps_csv=str(out_steps),
        out_runs_csv=str(out_runs),
        model_key=str(args.model_key),
        model_id=str(args.model_id),
        provider=provider,
        stimuli_json=str(args.stimuli_json),
        test_id=str(args.test_id),
        order_id=int(args.order_id),
        task_seed=int(args.task_seed),
        store_raw_response=bool(args.store_raw_response),
    )

    print(f"Done.\nSteps: {out_steps}\nRuns:  {out_runs}")


if __name__ == "__main__":
    main()
