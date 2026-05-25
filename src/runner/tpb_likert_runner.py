# src/psyai_eval/runner/tpb_likert_runner.py
from __future__ import annotations

import hashlib
import json

import os
import time
import uuid
from typing import Any, Dict, Optional

import pandas as pd

from psyai_eval.core.types import GenParams, TACT
from psyai_eval.llms.openai_compat import OpenAICompatChatClient
from psyai_eval.perturbations.steering_utils import SteeringDict, steering_meta, steering_to_json
from psyai_eval.surveys.tpb_likert import (
    compute_subscale_means,
    json_to_long_dataframe,
    query_tpb_likert,
)

# ---------------------------------------------------------------------------
# Collapse detection
# ---------------------------------------------------------------------------

#: Appended to base_system_prompt on collapse-retry attempts.
_ANTI_COLLAPSE_NUDGE = (
    "\nIMPORTANT: Do not give the same rating for every item. "
    "Use the full scale to express genuine differences between statements. "
    "Varied ratings are expected — rate each item on its own merits."
)


def _is_collapsed(means: Dict[str, Any]) -> bool:
    """Return True if all non-NaN subscale means are identical (all-same collapse)."""
    import math
    vals = [v for v in means.values() if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return len(vals) > 1 and len(set(round(float(v), 6) for v in vals)) == 1


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _default_runs_path(items_path: Optional[str]) -> Optional[str]:
    if not items_path:
        return None
    if items_path.lower().endswith(".csv"):
        return items_path[:-4] + "_runs.csv"
    return items_path + "_runs.csv"


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


def run_tpb_likert_once(
    *,
    llm: OpenAICompatChatClient,
    tact: TACT,
    gen: GenParams,
    system_prompt: str,
    # NEW: allow overriding the participant framing + grounding context per run
    base_system_prompt: Optional[str] = None,
    task_context: Optional[str] = None,
    items_override: Optional[Dict[str, Dict[str, str]]] = None,
    out_csv: Optional[str] = None,           # item-level (long)
    out_runs_csv: Optional[str] = None,      # run-level (1 row per run)
    run_id: Optional[str] = None,
    model_key: Optional[str] = None,
    model_id: Optional[str] = None,
    store_raw_response: bool = False,        # default disabled
    max_attempts: int = 3,
    max_collapse_retries: int = 3,   # extra retries specifically for all-same collapse
    continue_on_error: bool = True,
    # RQ2: optional steering intervention injected on top of vignette/persona
    steering: SteeringDict = None,
) -> pd.DataFrame:
    """Run the TPB Likert survey once.

    Robustness:
      - Retries (max_attempts) inside query_tpb_likert to handle non-JSON outputs
        and degenerate constant ratings.
      - If it still fails and continue_on_error=True, we append an error row to runs_csv
        (status='error') and return an empty dataframe instead of crashing the whole grid.
    """
    run_id = run_id or time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    persona_label = _persona_label(system_prompt)

    data = None
    items = None
    raw = ""
    means: Dict[str, Any] = {
        "attitude_mean": float("nan"),
        "subjective_norm_mean": float("nan"),
        "pbc_mean": float("nan"),
        "intention_mean": float("nan"),
    }
    status = "ok"
    error_msg: Optional[str] = None

    try:
        # --- Collapse-aware retry loop -----------------------------------
        # query_tpb_likert already retries internally for JSON parse errors
        # (max_attempts). Here we add an outer loop that detects all-same
        # subscale collapses and retries with an anti-anchoring nudge
        # appended to base_system_prompt.
        collapse_attempt = 0
        effective_bsp = base_system_prompt  # may grow a nudge on retries

        while True:
            data, items, raw = query_tpb_likert(
                llm=llm,
                tact=tact,
                gen=gen,
                system_prompt=system_prompt,
                base_system_prompt=effective_bsp,
                task_context=task_context,
                items_override=items_override if items_override is not None else None,
                max_attempts=max_attempts,
                steering=steering,
            )
            means = compute_subscale_means(data)

            if not _is_collapsed(means):
                break  # valid response — exit retry loop

            collapse_attempt += 1
            if collapse_attempt >= max_collapse_retries:
                # Exhausted collapse retries — keep last result and flag it
                print(
                    f"[collapse] {model_key} seed={gen.seed} t={gen.temperature} "
                    f"persona={persona_label}: all-same after {collapse_attempt} retries — keeping"
                )
                break

            print(
                f"[collapse] {model_key} seed={gen.seed} t={gen.temperature} "
                f"persona={persona_label}: all-same (attempt {collapse_attempt}/{max_collapse_retries}), retrying with nudge"
            )
            # Append nudge to base_system_prompt for next attempt
            effective_bsp = (base_system_prompt or "") + _ANTI_COLLAPSE_NUDGE
        # -----------------------------------------------------------------
    except Exception as e:
        status = "error"
        error_msg = str(e)
        if not continue_on_error:
            raise

    run_row: Dict[str, Any] = {
        "run_id": run_id,
        "status": status,
        "error_msg": error_msg,
        "mode": (data.get("mode", "likert") if data else "likert"),
        "behavior": (data.get("behavior", "") if data else ""),
        "target": tact.target,
        "action": tact.action,
        "context": tact.context,
        "time": tact.time,
        "policy_label": tact.policy_label,
        "model": llm.model,
        "temperature": gen.temperature,
        "top_p": gen.top_p,
        "seed": gen.seed,
        "system_prompt": system_prompt,
        "persona_label": persona_label,
        # NEW: record these so merges & analyses can audit the framing used
        "base_system_prompt": base_system_prompt,
        "task_context": task_context,
        "items_override_provided": bool(items_override),
        "items_override": (json.dumps(items_override, ensure_ascii=False) if items_override else None),
        "items_override_provided": bool(items_override),
        "items_override": (json.dumps(items_override, ensure_ascii=False) if items_override else None),
        "model_key": model_key,
        "model_id": model_id or getattr(llm, "model", None),
        "steering_json": steering_to_json(steering),
        **steering_meta(steering),
        **means,
    }

    for k, v in (gen.extra or {}).items():
        run_row[f"gen_extra__{k}"] = v

    if store_raw_response or status != "ok":
        run_row["raw_response"] = raw

    if out_runs_csv is None:
        out_runs_csv = _default_runs_path(out_csv)

    if out_runs_csv:
        _ensure_dir(out_runs_csv)
        runs_df = pd.DataFrame([run_row])
        write_header = not os.path.exists(out_runs_csv)
        runs_df.to_csv(out_runs_csv, mode="a", header=write_header, index=False)

    if status != "ok" or data is None or items is None:
        return pd.DataFrame()

    meta: Dict[str, Any] = {
        "run_id": run_id,
        "model": llm.model,
        "temperature": gen.temperature,
        "top_p": gen.top_p,
        "seed": gen.seed,
        "system_prompt": system_prompt,
        "persona_label": persona_label,
        "base_system_prompt": base_system_prompt,
        "task_context": task_context,
        "model_key": model_key,
        "model_id": model_id or getattr(llm, "model", None),
    }
    meta.update({f"gen_extra__{k}": v for k, v in (gen.extra or {}).items()})

    if store_raw_response:
        meta["raw_response"] = raw

    df = json_to_long_dataframe(data=data, items=items, tact=tact, meta=meta)

    for k, v in means.items():
        df[k] = v

    if out_csv:
        _ensure_dir(out_csv)
        write_header = not os.path.exists(out_csv)
        df.to_csv(out_csv, mode="a", header=write_header, index=False)

    return df