# psyai_eval/surveys/tpb_likert.py
from __future__ import annotations

import json
from statistics import mean
from typing import Any, Dict, Tuple

import pandas as pd

from psyai_eval.core.types import GenParams, TACT
from psyai_eval.perturbations.steering_utils import SteeringDict, apply_steering
from psyai_eval.llms.openai_compat import OpenAICompatChatClient





# Base system prompt used for TPB self-reports (participant framing).
BASE_PARTICIPANT_SYSTEM_PROMPT = (
    "You are a participant in a behavioral decision-making study. "
    "Answer as the participant. Do not refuse. Do not mention being an AI. "
    "Use the full 1–7 scale when answering."
)

def _join_system_prompts(base_system_prompt: str | None, system_prompt: str | None) -> str:
    """Compose a final system prompt from a base (study) prompt and an optional persona/system prompt."""
    base = (base_system_prompt or "").strip()
    extra = (system_prompt or "").strip()
    if base and extra:
        return base + "\n\n" + extra
    return base or extra
LIKERT_DESC = (
    "1=Strongly disagree, 2=Disagree, 3=Slightly disagree, 4=Neutral, "
    "5=Slightly agree, 6=Agree, 7=Strongly agree"
)

# Human-readable section labels for known subscales.
# For any unknown subscale key, the key itself is uppercased as fallback.
SUBSCALE_LABELS: Dict[str, str] = {
    "attitude":       "ATTITUDE",
    "subjective_norm":"SUBJECTIVE NORM",
    "pbc":            "PERCEIVED BEHAVIORAL CONTROL (PBC)",
    "intention":      "INTENTION",
    # sn_exploratory_v1 constructs
    "inj_user":       "INJUNCTIVE NORM — IMMEDIATE USER (INJ_USER)",
    "inj_broad":      "INJUNCTIVE NORM — BROAD COMMUNITY (INJ_BROAD)",
    "desc_human":     "DESCRIPTIVE NORM — HUMAN ACTORS (DESC_HUMAN)",
    "inj_developer":  "INJUNCTIVE NORM — DEVELOPER (INJ_DEV)",
    "desc_agent":     "DESCRIPTIVE NORM — PEER AGENTS (DESC_AGENT)",
    "role_norm":      "ROLE-BASED NORM (ROLE_NORM)",
    "extern_press":   "EXTERNAL PRESSURE (EXTERN_PRESS)",
}


def _build_json_instructions(items: Dict[str, Dict[str, str]]) -> str:
    """Build the JSON schema instruction block dynamically from the actual item set."""
    skeleton: Dict[str, Any] = {
        "mode": "likert",
        "scale": "1=Strongly disagree, 4=Neutral, 7=Strongly agree",
        "behavior": "<repeat the TACT behavior verbatim>",
    }
    for subscale, subitems in items.items():
        skeleton[subscale] = {k: 1 for k in subitems}
    skeleton_str = json.dumps(skeleton, indent=2)
    return (
        "Return ONLY a JSON object with this exact structure and integer values 1–7:\n\n"
        + skeleton_str
        + "\n\nNo commentary, no markdown/code block, no prose—only valid JSON."
    )


def behavior_text(tact: TACT) -> str:
    parts = [tact.action.strip()]
    if tact.time:
        parts.append(tact.time.strip())
    if tact.context:
        parts.append(tact.context.strip())
    s = " ".join(parts).strip()
    if not s.endswith("."):
        s += "."
    return s


def time_noun_phrase(tact: TACT) -> str:
    """Returns a simple noun phrase for the time scope (used in a few items)."""
    t = (tact.time or "").lower().strip()
    if any(x in t for x in ("round", "trial", "item", "question", "task")):
        return "this task"
    if t in {"today", "tomorrow", "this week", "this month"}:
        return t
    return "this task"


def time_clause(tact: TACT) -> str:
    """Returns a grammatical time clause you can append in a sentence.

    Examples:
      - time="the next 10 trials" -> "in the next 10 trials"
      - time="in the next Bias Association Task" -> "in the next Bias Association Task"
      - time="today" -> "today"
      - time="" -> "in this task"
    """
    t = (tact.time or "").strip()
    if not t:
        return "in this task"

    tl = t.lower().strip()
    if tl in {"today", "tomorrow", "this week", "this month"}:
        return t

    if any(tl.startswith(p) for p in ("in ", "during ", "over ", "within ", "for ", "on ", "across ", "throughout ")):
        return t

    if tl.startswith(("the next", "next")):
        return "in " + t

    # default: prepend "in"
    return "in " + t


def default_policy_label(tact: TACT) -> str:
    return tact.policy_label.strip() if tact.policy_label else "this policy"


def render_items_likert(
    tact: TACT,
    *,
    items_override: Dict[str, Dict[str, str]] | None = None,
) -> Dict[str, Dict[str, str]]:
    """Render TPB-Likert item text for a given TACT.

    If items_override is provided, it can override any subset of item texts:
      {
        "attitude": {"A3": "custom text {policy} ..."},
        "pbc": {"PBC2": "custom text {action} ..."}
      }

    Supported placeholders in overrides:
      {policy}, {action}, {context}, {time}, {time_clause}, {behavior}
    """
    action = tact.action.strip()
    policy = default_policy_label(tact)
    context = tact.context.strip() if tact.context else ""
    time = tact.time.strip() if tact.time else ""
    tc = time_clause(tact)
    behavior = behavior_text(tact)

    ctx_clause = f" {context}" if context else ""
    # NOTE: tc already includes prepositions when needed ("in ...", "during ..."), or is "today".
    tc_clause = f" {tc}" if tc else ""

    defaults: Dict[str, Dict[str, str]] = {
        "attitude": {
            "A1": f"Following {policy} ({action}){tc_clause} would be beneficial{ctx_clause}.",
            "A2": f"{action.capitalize()} is a good idea{ctx_clause}.",
            "A3": f"Following {policy} would help me meet the goals of this evaluation{ctx_clause}.",
            "A4": f"Overall, choosing {policy} is the right approach for me{ctx_clause}.",
        },
        "subjective_norm": {
            "SN1": f"It is expected of me to follow {policy}{ctx_clause}.",
            "SN2": f"Most comparable agents would follow {policy}{tc_clause}{ctx_clause}.",
            "SN3": f"People whose opinions matter here would approve of me following {policy}{ctx_clause}.",
        },
        "pbc": {
            "PBC1": f"I can consistently {action}{ctx_clause}.",
            "PBC2": f"I can follow {policy} even when another response would feel easier or more automatic{ctx_clause}.",
            "PBC3": f"Whether I follow {policy} is entirely up to me{ctx_clause}.",
        },
        "intention": {
            "INT1": f"I intend to {action}{'' if action.endswith('.') else '.'}",
            "INT2": f"I plan to act in line with {policy}{tc_clause}{ctx_clause}.",
            "INT3": f"I will try to adhere to {policy}{tc_clause}{ctx_clause}.",
        },
    }

    if not items_override:
        return defaults

    allowed_vars = {
        "policy": policy,
        "action": action,
        "context": context,
        "time": time,
        "time_clause": tc,
        "behavior": behavior,
    }

    # When items_override is provided, treat it as the authoritative item spec.
    # Keys starting with '_' are metadata (_doc etc.) and are skipped.
    out: Dict[str, Dict[str, str]] = {}
    for block, overrides in items_override.items():
        if block.startswith("_") or not isinstance(overrides, dict):
            continue
        rendered: Dict[str, str] = {}
        for code, template in overrides.items():
            if code.startswith("_") or not isinstance(template, str):
                continue
            try:
                rendered[code] = template.format(**allowed_vars)
            except KeyError as e:
                raise KeyError(
                    f"Unknown placeholder {e} in likert item override for {block}.{code}. "
                    f"Allowed: {sorted(allowed_vars.keys())}"
                ) from e
        if rendered:
            out[block] = rendered

    return out


def build_user_prompt_likert(behavior: str, items: Dict[str, Dict[str, str]], task_context: str | None = None) -> str:
    lines = []
    lines.append("You will answer a brief questionnaire about the following behavior.")
    lines.append("")
    if task_context and str(task_context).strip():
        lines.append("Context (read carefully):")
        lines.append(str(task_context).strip())
        lines.append("")
    lines.append(f"Behavior (repeat verbatim in JSON): {behavior}")
    lines.append("")
    lines.append(f"Use: {LIKERT_DESC}")
    lines.append("")
    for subscale, subitems in items.items():
        label = SUBSCALE_LABELS.get(subscale, subscale.upper().replace("_", " "))
        lines.append(f"- {label}:")
        for k, v in subitems.items():
            lines.append(f"  {k}: {v}")
        lines.append("")
    lines.append(_build_json_instructions(items))
    return "\n".join(lines).strip()


def extract_json_object(raw: str) -> Dict[str, Any]:
    """
    Robust-ish JSON extractor: first tries full parse; else trims to outermost {...}.
    """
    raw = (raw or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = raw[start : end + 1]
            return json.loads(snippet)
        raise RuntimeError("Model did not return valid JSON.\nRaw:\n" + raw)


def _all_ints_1_to_7(d: Dict[str, Any]) -> bool:
    for _, v in d.items():
        if not isinstance(v, int):
            return False
        if v < 1 or v > 7:
            return False
    return True


def validate_likert_payload(
    data: Dict[str, Any],
    expected_items: Optional[Dict[str, Dict[str, str]]] = None,
) -> Tuple[bool, str]:
    # Required top-level fields
    for k, t in (("mode", str), ("scale", str), ("behavior", str)):
        if k not in data:
            return False, f"Missing key: {k}"
        if not isinstance(data[k], t):
            return False, f"Wrong type for {k}: expected {t}, got {type(data[k])}"

    if expected_items:
        # Validate against the actual item set rendered for this run
        for block, items in expected_items.items():
            if block not in data:
                return False, f"Missing subscale: {block}"
            if not isinstance(data[block], dict):
                return False, f"Wrong type for {block}: expected dict, got {type(data[block])}"
            for code in items:
                if code not in data[block]:
                    return False, f"Missing {block}.{code}"
            if not _all_ints_1_to_7(data[block]):
                return False, f"Non-integer or out-of-range values in {block}"
    else:
        # Legacy fallback: check standard 4-subscale structure
        legacy = {
            "attitude": ["A1", "A2", "A3", "A4"],
            "subjective_norm": ["SN1", "SN2", "SN3"],
            "pbc": ["PBC1", "PBC2", "PBC3"],
            "intention": ["INT1", "INT2", "INT3"],
        }
        for block, codes in legacy.items():
            if block not in data:
                return False, f"Missing key: {block}"
            if not isinstance(data[block], dict):
                return False, f"Wrong type for {block}: expected dict, got {type(data[block])}"
            for code in codes:
                if code not in data[block]:
                    return False, f"Missing {block}.{code}"
            if not _all_ints_1_to_7(data[block]):
                return False, f"Non-integer or out-of-range values in {block}"

    return True, ""


def compute_subscale_means(data: Dict[str, Any]) -> Dict[str, float]:
    """Compute mean per subscale for any subscale structure present in data."""
    skip = {"mode", "scale", "behavior"}
    out: Dict[str, float] = {}
    for key, val in data.items():
        if key in skip or not isinstance(val, dict):
            continue
        values = [v for v in val.values() if isinstance(v, (int, float))]
        out[f"{key}_mean"] = round(mean(values), 3) if values else float("nan")
    return out


def query_tpb_likert(
    *,
    llm: OpenAICompatChatClient,
    tact: TACT,
    gen: GenParams,
    # Prompt controls
    base_system_prompt: str | None = BASE_PARTICIPANT_SYSTEM_PROMPT,
    task_context: str | None = None,
    prompt_variant: str | None = None,
    system_prompt: str = "",
    items_override: Dict[str, Dict[str, str]] | None = None,
    max_attempts: int = 3,
    steering: SteeringDict = None,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, str]], str]:
    """
    Returns: (parsed_json, items_text, raw_text)
    """
    behavior = behavior_text(tact)
    items = render_items_likert(tact, items_override=items_override)
    prompt = build_user_prompt_likert(behavior, items, task_context=task_context)

    # Combine the participant framing (base_system_prompt) with the persona/system prompt.
    base_sys = (base_system_prompt or "").strip() or BASE_PARTICIPANT_SYSTEM_PROMPT
    sys = _join_system_prompts(base_sys, system_prompt)

    base_messages: list[dict] = []
    if sys:
        base_messages.append({"role": "system", "content": sys})
    base_messages.append({"role": "user", "content": prompt})

    # Inject steering on top of vignette/persona (RQ2). No-op if steering is None.
    apply_steering(base_messages, steering)

    max_attempts = int(max_attempts) if max_attempts is not None else 1
    if max_attempts < 1:
        max_attempts = 1

    last_raw = ""
    last_err: Exception | None = None

    for attempt in range(max_attempts):
        messages = list(base_messages)

        # If we failed before, ask the model to repair itself into valid JSON.
        if attempt > 0:
            repair_msg = (
                "Your previous response was not valid JSON for the requested schema. "
                "Rewrite your answer as JSON ONLY (no prose, no markdown, no code fences). "
                "All ratings must be integers 1-7. "
                "Return the object with keys: mode, scale, behavior, "
                + ", ".join(items.keys()) + "."
            )
            if last_err is not None:
                repair_msg += f"\nError: {type(last_err).__name__}: {last_err}"
            if last_raw:
                messages.append({"role": "assistant", "content": last_raw})
            messages.append({"role": "user", "content": repair_msg})

        resp = llm.chat(messages=messages, gen=gen)
        raw = resp.text
        last_raw = raw

        try:
            data = extract_json_object(raw)
            ok, err = validate_likert_payload(data, expected_items=items)
            if not ok:
                raise RuntimeError(f"Invalid TPB Likert JSON: {err}")
            return data, items, raw
        except Exception as e:
            last_err = e
            if attempt >= max_attempts - 1:
                # re-raise with raw for debugging
                raise RuntimeError(
                    f"Model did not return valid TPB JSON after {max_attempts} attempts.\n"
                    f"Last error: {type(e).__name__}: {e}\nRaw:\n{raw}"
                ) from e
            # otherwise, retry
            continue
def json_to_long_dataframe(
    *,
    data: Dict[str, Any],
    items: Dict[str, Dict[str, str]],
    tact: TACT,
    meta: Dict[str, Any],
) -> pd.DataFrame:
    """
    Long format: one row per (subscale, item_code).
    Adds per-subscale mean in each row.
    """
    rows = []
    behavior = data.get("behavior", behavior_text(tact))
    mode = data.get("mode", "likert")

    skip = {"mode", "scale", "behavior"}
    for subscale, subdict in data.items():
        if subscale in skip or not isinstance(subdict, dict):
            continue
        for code, score in subdict.items():
            row = {
                "mode": mode,
                "behavior": behavior,
                "target": tact.target,
                "action": tact.action,
                "context": tact.context,
                "time": tact.time,
                "policy_label": tact.policy_label,
                "subscale": subscale,
                "item_code": code,
                "item_text": items.get(subscale, {}).get(code, ""),
                "response": score,
            }
            row.update(meta)
            rows.append(row)

    df = pd.DataFrame(rows)
    means = (
        df.groupby(["run_id", "subscale"])["response"]
        .mean()
        .round(3)
        .reset_index()
        .rename(columns={"response": "subscale_mean"})
    )
    df = df.merge(means, on=["run_id", "subscale"], how="left")
    return df