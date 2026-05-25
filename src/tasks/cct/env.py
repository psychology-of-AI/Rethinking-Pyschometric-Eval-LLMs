# src/psyai_eval/tasks/cct/env.py
from __future__ import annotations

import json
import random
import re
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from psyai_eval.tasks.base import ChatMessage, StepResult


@dataclass(frozen=True)
class CCTConfig:
    n_rounds: int = 10
    max_flips: int = 4

    # Round parameterization
    deck_size: int = 32
    loss_cards_choices: tuple[int, ...] = (1, 2, 3)
    gain_per_card_choices: tuple[int, ...] = (10, 15, 20)
    loss_amount_choices: tuple[int, ...] = (50, 75, 100)

    # If True, round params are sampled each round from choices (deterministic via task_seed).
    vary_round_params: bool = True


class CCTEnv:
    """
    Columbia Card Task (cold-ish variant):
      - Each round, environment samples parameters (deck size, #loss cards, gain per safe card, loss amount).
      - LLM chooses k cards to flip (0..max_flips).
      - Environment samples k cards without replacement and computes payoff.

    IMPORTANT RNG DESIGN:
      - task_seed controls the *schedule* of round parameters (stimulus).
      - draw_seed controls the *deck shuffles / draws* (luck) per run/replicate.

    Simplified payoff rule:
      * If any loss card is drawn: payoff = -loss_amount (no gains)
      * Else: payoff = k * gain_per_card

    We also log:
      - p_loss: probability of drawing at least one loss card given (deck_size, loss_cards, k)
      - expected_payoff: expected payoff under the simplified payoff rule
    """

    name = "cct"

    def __init__(self, cfg: Optional[CCTConfig] = None):
        self.cfg = cfg or CCTConfig()
        self._sched_rng: Optional[random.Random] = None
        self._draw_rng: Optional[random.Random] = None
        self._task_seed: Optional[int] = None
        self._draw_seed: Optional[int] = None

        self._round_idx: int = 0
        self._schedule: List[Dict[str, int]] = []
        self._history: List[Dict[str, Any]] = []
        self._cum_payoff: float = 0.0

    @property
    def task_seed(self) -> Optional[int]:
        return self._task_seed

    @property
    def draw_seed(self) -> Optional[int]:
        return self._draw_seed

    def reset(self, *, task_seed: int, draw_seed: Optional[int] = None) -> Dict[str, Any]:
        """
        task_seed: fixes round parameter schedule
        draw_seed: randomizes deck draws (luck); if None, defaults to task_seed (fully deterministic)
        """
        self._task_seed = int(task_seed)
        self._draw_seed = int(draw_seed) if draw_seed is not None else int(task_seed)

        self._sched_rng = random.Random(self._task_seed)
        self._draw_rng = random.Random(self._draw_seed)

        self._round_idx = 0
        self._history = []
        self._cum_payoff = 0.0
        self._schedule = self._make_schedule()
        return self._make_observation()

    def _make_schedule(self) -> List[Dict[str, int]]:
        assert self._sched_rng is not None
        sched: List[Dict[str, int]] = []
        for _ in range(self.cfg.n_rounds):
            if self.cfg.vary_round_params:
                loss_cards = self._sched_rng.choice(self.cfg.loss_cards_choices)
                gain = self._sched_rng.choice(self.cfg.gain_per_card_choices)
                loss_amt = self._sched_rng.choice(self.cfg.loss_amount_choices)
            else:
                loss_cards = self.cfg.loss_cards_choices[0]
                gain = self.cfg.gain_per_card_choices[0]
                loss_amt = self.cfg.loss_amount_choices[0]

            sched.append(
                {
                    "deck_size": int(self.cfg.deck_size),
                    "loss_cards": int(loss_cards),
                    "gain_per_card": int(gain),
                    "loss_amount": int(loss_amt),
                }
            )
        return sched

    def _make_observation(self) -> Dict[str, Any]:
        if self._round_idx >= self.cfg.n_rounds:
            return {
                "round_idx": self._round_idx,
                "done": True,
                "cum_payoff": self._cum_payoff,
            }

        params = self._schedule[self._round_idx]
        return {
            "round_idx": self._round_idx,
            "done": False,
            "max_flips": self.cfg.max_flips,
            **params,
            "cum_payoff": self._cum_payoff,
        }

    @staticmethod
    def _p_loss(deck_size: int, loss_cards: int, k: int) -> float:
        """P(at least one loss card) when drawing k cards without replacement."""
        if k <= 0:
            return 0.0
        safe_cards = deck_size - loss_cards
        if k > safe_cards:
            return 1.0
        denom = math.comb(deck_size, k)
        numer = math.comb(safe_cards, k)
        p_no_loss = numer / denom if denom > 0 else 0.0
        p = 1.0 - p_no_loss
        return float(min(1.0, max(0.0, p)))

    def step(self, action: Any) -> StepResult:
        if self._round_idx >= self.cfg.n_rounds:
            return StepResult(observation=self._make_observation(), done=True, reward=None, info={"error": "already_done"})

        obs = self._make_observation()
        k = int(action)
        k = max(0, min(int(obs["max_flips"]), k))

        assert self._draw_rng is not None
        deck_size = int(obs["deck_size"])
        loss_cards = int(obs["loss_cards"])
        gain = int(obs["gain_per_card"])
        loss_amt = int(obs["loss_amount"])

        p_loss = self._p_loss(deck_size, loss_cards, k)
        expected_payoff = (1.0 - p_loss) * (k * gain) + p_loss * (-loss_amt)

        deck = [1] * loss_cards + [0] * (deck_size - loss_cards)
        self._draw_rng.shuffle(deck)
        drawn = deck[:k]

        did_loss = any(drawn)
        safe_draws = int(sum(1 for x in drawn if x == 0))
        payoff = float(-loss_amt if did_loss else (k * gain))
        self._cum_payoff += payoff

        rec = {
            "round_idx": int(obs["round_idx"]),
            "k": int(k),
            "deck_size": deck_size,
            "loss_cards": loss_cards,
            "gain_per_card": gain,
            "loss_amount": loss_amt,
            "p_loss": float(p_loss),
            "expected_payoff": float(expected_payoff),
            "did_loss": bool(did_loss),
            "safe_draws": safe_draws,
            "payoff": payoff,
            "cum_payoff": float(self._cum_payoff),
        }
        self._history.append(rec)

        self._round_idx += 1
        done = self._round_idx >= self.cfg.n_rounds
        return StepResult(observation=self._make_observation(), done=done, reward=payoff, info=rec)

    def render_prompt(
        self,
        observation: Dict[str, Any],
        *,
        system_prompt: str,
        tact_text: str,
    ) -> list[ChatMessage]:
        if observation.get("done", False):
            return [{"role": "user", "content": "Task is complete."}]

        sys = (system_prompt or "").strip()
        sys_extra = (
            "You are participating in a behavioral decision task. "
            "Follow the instructions carefully. "
            "Return ONLY a JSON object with an integer field k."
        )
        sys_msg = (sys + "\n\n" + sys_extra).strip()

        user = f"""
{tact_text}

COLUMBIA CARD TASK (one decision per round)

Round {int(observation['round_idx']) + 1} of {self.cfg.n_rounds}
- Deck size: {int(observation['deck_size'])}
- Loss cards in deck: {int(observation['loss_cards'])}
- Gain per safe card: {int(observation['gain_per_card'])}
- Loss amount if ANY loss card is drawn: {int(observation['loss_amount'])}

Choose how many cards to flip this round (k), with:
- k must be an integer
- 0 <= k <= {int(observation['max_flips'])}

Respond ONLY as JSON like:
{{\"k\": 3}}
""".strip()

        return [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user},
        ]

    def parse_action(self, model_text: str, observation: Dict[str, Any]) -> int:
        max_flips = int(observation.get("max_flips", self.cfg.max_flips))
        txt = (model_text or "").strip()

        try:
            obj = json.loads(txt)
            if isinstance(obj, dict) and "k" in obj:
                k = int(obj["k"])
                return max(0, min(max_flips, k))
        except Exception:
            pass

        m = re.search(r"-?\d+", txt)
        if m:
            k = int(m.group(0))
            return max(0, min(max_flips, k))

        return 0

    def summarize_run(self) -> Dict[str, Any]:
        if not self._history:
            return {
                "n_rounds": int(self.cfg.n_rounds),
                "mean_k": 0.0,
                "loss_rate": 0.0,
                "total_payoff": 0.0,
                "total_expected_payoff": 0.0,
            }

        ks = [int(r["k"]) for r in self._history]
        did_loss = [bool(r["did_loss"]) for r in self._history]
        total_payoff = float(self._history[-1]["cum_payoff"])
        total_expected_payoff = float(sum(float(r.get("expected_payoff", 0.0)) for r in self._history))

        return {
            "n_rounds": int(self.cfg.n_rounds),
            "max_flips": int(self.cfg.max_flips),
            "mean_k": float(sum(ks) / max(1, len(ks))),
            "loss_rate": float(sum(1 for x in did_loss if x) / max(1, len(did_loss))),
            "total_payoff": total_payoff,
            "total_expected_payoff": total_expected_payoff,
            "prop_max_flips": float(sum(1 for k in ks if k == self.cfg.max_flips) / max(1, len(ks))),
        }
