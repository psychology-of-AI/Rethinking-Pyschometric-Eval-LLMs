# src/psyai_eval/tasks/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple


ChatMessage = Dict[str, str]  # {"role": "system"|"user"|"assistant", "content": "..."}


@dataclass(frozen=True)
class StepResult:
    observation: Dict[str, Any]
    done: bool
    reward: Optional[float] = None
    info: Dict[str, Any] | None = None


class Task(Protocol):
    """Minimal interface for stateful behavioral tasks."""

    name: str

    def reset(self, *, task_seed: int) -> Dict[str, Any]:
        ...

    def step(self, action: Any) -> StepResult:
        ...

    def render_prompt(
        self,
        observation: Dict[str, Any],
        *,
        system_prompt: str,
        tact_text: str,
    ) -> list[ChatMessage]:
        ...

    def parse_action(self, model_text: str, observation: Dict[str, Any]) -> Any:
        ...

    def summarize_run(self) -> Dict[str, Any]:
        ...
