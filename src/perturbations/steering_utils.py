# psyai_eval/perturbations/steering_utils.py
"""
Steering injection utility.

Steering is a structured perturbation applied ON TOP of the vignette/persona
(which is the RQ1 sampling device). It is the treatment for RQ2 (causal analysis).

Supported steering types (steering["type"]):
  - "direct_policy"   : explicit behavioral instruction ("Avoid risk at all costs.")
  - "tpb_persuasive"  : targets a specific TPB construct with an indirect persuasive message
  - "persona"         : a situated identity vignette (same format as RQ1 vignettes, but
                        chosen for directional intent rather than factorial sampling)
  - <future>          : e.g. "big5", "emotion_prime", "role" — add new types here only;
                        runner code does not need to change.

Supported injection sites (steering["injection_site"]):
  - "system"       : appended to the composed system message (default)
  - "user_prefix"  : prepended to the last user message
  - "user_suffix"  : appended to the last user message

Schema of the steering dict (all fields optional except "type" and "content"):
  {
    "type":           str,              # steering type label (for logging)
    "direction":      str | None,       # behavioral direction label (e.g. "loss_averse")
    "tpb_construct":  str | None,       # for tpb_persuasive: "attitude" | "subjective_norm" | "pbc"
    "content":        str,              # the text to inject
    "injection_site": str,              # "system" | "user_prefix" | "user_suffix"
    "notes":          str | None        # human-readable design notes (not injected)
  }

Usage:
    from psyai_eval.perturbations.steering_utils import apply_steering

    # After building messages list:
    messages = apply_steering(messages, steering=v.get("steering"))
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


# Type alias for clarity throughout the codebase.
SteeringDict = Optional[Dict[str, Any]]

_VALID_SITES = {"system", "user_prefix", "user_suffix"}
_VALID_TYPES = {"direct_policy", "tpb_persuasive", "persona"}  # extensible — not enforced as error


def apply_steering(
    messages: List[Dict[str, str]],
    steering: SteeringDict,
) -> List[Dict[str, str]]:
    """
    Inject steering content into a messages list in-place and return it.

    If steering is None, empty, or has no content, messages are returned unchanged.
    Raises ValueError for an unrecognised injection_site.

    Args:
        messages : list of {"role": ..., "content": ...} dicts, as passed to the LLM.
                   Must have at least one message. System message (if present) must be first.
        steering : dict conforming to the schema above, or None.

    Returns:
        The (mutated) messages list.
    """
    if not steering:
        return messages

    content = (steering.get("content") or "").strip()
    if not content:
        return messages

    site = (steering.get("injection_site") or "system").strip().lower()
    if site not in _VALID_SITES:
        raise ValueError(
            f"steering.injection_site must be one of {sorted(_VALID_SITES)}, got {site!r}"
        )

    if not messages:
        # No messages to inject into — create a system message.
        messages.append({"role": "system", "content": content})
        return messages

    if site == "system":
        # Append to system message if one exists; otherwise prepend a new system message.
        if messages[0].get("role") == "system":
            existing = (messages[0].get("content") or "").strip()
            messages[0]["content"] = (existing + "\n\n" + content).strip()
        else:
            messages.insert(0, {"role": "system", "content": content})

    elif site == "user_prefix":
        # Prepend to the last user-turn message.
        last_user_idx = _last_user_idx(messages)
        existing = (messages[last_user_idx].get("content") or "").strip()
        messages[last_user_idx]["content"] = (content + "\n\n" + existing).strip()

    elif site == "user_suffix":
        # Append to the last user-turn message.
        last_user_idx = _last_user_idx(messages)
        existing = (messages[last_user_idx].get("content") or "").strip()
        messages[last_user_idx]["content"] = (existing + "\n\n" + content).strip()

    return messages


def steering_meta(steering: SteeringDict) -> Dict[str, Any]:
    """
    Extract loggable metadata from a steering dict for inclusion in run_row CSVs.
    Returns an empty dict if steering is None.
    """
    if not steering:
        return {
            "steering_type": None,
            "steering_direction": None,
            "steering_tpb_construct": None,
            "steering_injection_site": None,
            "steering_content": None,
        }
    return {
        "steering_type":          steering.get("type"),
        "steering_direction":     steering.get("direction"),
        "steering_tpb_construct": steering.get("tpb_construct"),
        "steering_injection_site": steering.get("injection_site", "system"),
        "steering_content":       (steering.get("content") or "").strip() or None,
    }


def steering_to_json(steering: SteeringDict) -> Optional[str]:
    """Serialise a steering dict to JSON string for CSV storage. Returns None if no steering."""
    if not steering:
        return None
    return json.dumps(
        {k: v for k, v in steering.items() if k != "notes"},
        ensure_ascii=False,
    )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _last_user_idx(messages: List[Dict[str, str]]) -> int:
    """Return the index of the last message with role='user'. Falls back to last message."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return len(messages) - 1