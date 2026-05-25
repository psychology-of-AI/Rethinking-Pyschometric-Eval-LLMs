# psyai_eval/llms/openai_compat.py
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, List, Optional

from openai import OpenAI

from psyai_eval.core.types import GenParams


@dataclass
class LLMResponse:
    text: str
    raw: Any


class OpenAICompatChatClient:
    """
    Minimal wrapper for OpenAI-compatible chat APIs.

    Works with llama-api.com via:
      OpenAI(api_key=..., base_url="https://api.llama-api.com")

    Works with OpenRouter via:
      OpenAI(api_key=..., base_url="https://openrouter.ai/api/v1")
    """

    # Keys that our code may attach to GenParams.extra for bookkeeping,
    # but which must NEVER be forwarded to the provider API.
    _INTERNAL_EXTRA_KEYS = {
        "prompt_variant",
        "prompt_variant_id",
        "variant_id",
        "sweep_id",
        "grid_id",
        "run_id",
        "condition_id",
        "replicate_index",
        "note",
        "survey_mode"
    }

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        default_headers: Optional[Dict[str, str]] = None,
    ):
        # Some OpenAI SDK versions accept default_headers; keep a safe fallback.
        try:
            self.client = OpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers or None)
        except TypeError:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.base_url = base_url

        #print("DEBUG OpenAICompatChatClient base_url=", base_url, "key_prefix=", api_key[:15])

    def _to_dict(self, obj: Any) -> Any:
        """Best-effort conversion of SDK objects to plain dict for debugging."""
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj
        for meth in ("model_dump", "dict"):
            fn = getattr(obj, meth, None)
            if callable(fn):
                try:
                    return fn()
                except Exception:
                    pass
        return str(obj)

    def _extract_error_message(self, resp: Any) -> str:
        """
        OpenRouter can return an OpenAI-ish payload with top-level `error`
        while `choices` is null/None (often provider failure).
        """
        d = self._to_dict(resp)
        if isinstance(d, dict):
            err = d.get("error") or d.get("provider_error")
            if isinstance(err, dict):
                msg = err.get("message") or err.get("error")
                if msg:
                    return str(msg)
                return json.dumps(err, ensure_ascii=False)[:2000]
            if err is not None:
                return str(err)
        return ""
    
    def _filter_extra(self, extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Drop internal/metadata keys (and obvious private keys) from GenParams.extra.
        This prevents accidental passing of non-API kwargs (e.g., prompt_variant)
        into chat.completions.create(...).
        """
        if not extra:
            return {}

        out: Dict[str, Any] = {}
        for k, v in extra.items():
            if k in self._INTERNAL_EXTRA_KEYS:
                continue
            if isinstance(k, str) and k.startswith("_"):
                continue
            out[k] = v
        return out

    def chat(
        self,
        *,
        messages: List[Dict[str, str]],
        gen: GenParams,
    ) -> LLMResponse:
        # Build kwargs carefully; some providers reject unsupported fields.
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": gen.temperature,
            "top_p": gen.top_p,
            "max_tokens": gen.max_tokens,
        }

        # seed/top_k etc. may or may not be supported by a given OpenAI-compatible endpoint.
        if gen.seed is not None and "openrouter.ai" not in (self.base_url or "").lower():
            kwargs["seed"] = gen.seed

        extra = self._filter_extra(gen.extra)
        if extra:
            # Allow extra to set things like response_format (json mode), stop, penalties, etc.
            # But never allow it to override core keys.
            for k, v in extra.items():
                if k in kwargs:
                    continue
                kwargs[k] = v

        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            # If provider rejects unsupported params (commonly 'seed'), retry once without them.
            msg = str(e).lower()
            if ("seed" in msg and "unsupported" in msg) or ("unknown" in msg and "seed" in msg):
                kwargs.pop("seed", None)
                resp = self.client.chat.completions.create(**kwargs)
            else:
                raise

        # Robust handling: some OpenAI-compatible providers (incl. OpenRouter routing)
        # can return an error payload where choices is None/empty.
        choices = getattr(resp, "choices", None)
        if not choices:
            err_msg = self._extract_error_message(resp)
            payload = self._to_dict(resp)
            if err_msg:
                raise RuntimeError(
                    f"LLM call returned no choices (model={self.model}, base_url={self.base_url}). "
                    f"Provider error: {err_msg}"
                )
            raise RuntimeError(
                f"LLM call returned no choices (model={self.model}, base_url={self.base_url}). "
                f"Raw payload (truncated): {str(payload)[:2000]}"
            )

        msg = choices[0].message
        text = (getattr(msg, "content", None) or "")
        return LLMResponse(text=text, raw=resp)
