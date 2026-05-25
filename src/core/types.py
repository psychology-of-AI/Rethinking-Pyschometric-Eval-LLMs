# psyai_eval/core/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TACT:
    target: str
    action: str
    context: str
    time: str
    policy_label: Optional[str] = None


@dataclass(frozen=True)
class GenParams:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 700
    seed: Optional[int] = None

    # For OpenAI-compatible providers that accept non-standard knobs (e.g., top_k)
    extra: Dict[str, Any] = field(default_factory=dict)
