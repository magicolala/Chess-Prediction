"""Hugging Face Inference API-backed sequence provider."""

from __future__ import annotations

import math
import os
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .base import SequenceProvider

try:  # pragma: no cover - optional dependency import
    from huggingface_hub import InferenceClient
except Exception:  # pragma: no cover - handled lazily
    InferenceClient = None  # type: ignore[misc]


def _get_attr(mapping: object, name: str, default=None):
    """Return attribute or key value from the mapping-like object."""

    if mapping is None:
        return default
    if isinstance(mapping, dict):
        return mapping.get(name, default)
    return getattr(mapping, name, default)


def _normalize_token_entry(entry: object) -> Optional[Tuple[str, float]]:
    """Normalize a Hugging Face token entry into ``(text, logprob)``."""

    if entry is None:
        return None
    text = _get_attr(entry, "token")
    if text is None:
        text = _get_attr(entry, "text")
    if text is None:
        token_id = _get_attr(entry, "id")
        if token_id is not None:
            text = str(token_id)
    logprob = _get_attr(entry, "logprob")
    if logprob is None:
        prob = _get_attr(entry, "prob")
        if prob not in (None, 0):
            try:
                logprob = math.log(float(prob))
            except (TypeError, ValueError):
                logprob = None
    if text is None or logprob is None:
        return None
    try:
        logprob_value = float(logprob)
    except (TypeError, ValueError):
        return None
    token_text = str(text).replace("▁", " ").strip()
    if not token_text:
        return None
    return token_text, logprob_value


def _unique_preserve_order(pairs: Iterable[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Deduplicate entries while preserving first occurrence order."""

    seen: Dict[str, float] = {}
    ordered: List[Tuple[str, float]] = []
    for text, logprob in pairs:
        if text in seen:
            continue
        seen[text] = logprob
        ordered.append((text, logprob))
    return ordered


def _extract_status_code(exc: Exception) -> Optional[int]:
    """Try to recover an HTTP status code from an exception."""

    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    if response is not None:
        code = getattr(response, "status_code", None)
        if isinstance(code, int):
            return code
    request = getattr(exc, "request", None)
    if request is not None:
        code = getattr(request, "status_code", None)
        if isinstance(code, int):
            return code
    return None


class HuggingFaceServerlessProvider(SequenceProvider):
    """Sequence provider querying Hugging Face serverless text generation."""

    def __init__(
        self,
        *,
        model_id: str,
        api_token: str | None = None,
        client=None,
        top_n_tokens: int = 10,
        temperature: float = 0.0,
        do_sample: Optional[bool] = None,
        expose_probs: bool | None = None,
        max_retries: int = 3,
        retry_base_delay: float = 0.5,
        rate_limit_delay: float = 1.0,
        sleep: Optional[Callable[[float], None]] = None,
    ) -> None:
        if client is None:
            if InferenceClient is None:  # pragma: no cover - dependency guard
                raise ImportError(
                    "huggingface-hub must be installed to use HuggingFaceServerlessProvider"
                )
            client = InferenceClient(model=model_id, token=api_token or None)
        self.client = client
        self.model_id = model_id
        self.api_token = api_token or ""
        self.top_n_tokens = max(1, int(top_n_tokens or 1))
        self.temperature = float(temperature)
        self.do_sample = bool(do_sample) if do_sample is not None else self.temperature > 0
        self.expose_probs = bool(expose_probs) if expose_probs is not None else False
        self.max_retries = max(1, int(max_retries or 1))
        self.retry_base_delay = max(0.0, float(retry_base_delay))
        self.rate_limit_delay = max(0.0, float(rate_limit_delay))
        self._sleep = sleep if sleep is not None else time.sleep

    # SequenceProvider protocol -------------------------------------------------
    def get_top_sequences(
        self,
        prompt: str,
        legal_moves: List[str],
        depth: int,
        prob_threshold: float,
        top_k: int,
    ) -> List[Tuple[str, float]]:
        if top_k <= 0:
            return []

        request_top_n = max(self.top_n_tokens, int(top_k))
        response = self._call_with_retries(prompt, request_top_n)
        return self._parse_response(response, top_k)

    # Internal helpers ---------------------------------------------------------
    def _call_with_retries(self, prompt: str, top_n: int):
        attempts = 0
        last_exc: Optional[Exception] = None
        while attempts < self.max_retries:
            try:
                return self.client.text_generation(
                    prompt,
                    max_new_tokens=1,
                    details=True,
                    return_full_text=False,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                    top_n_tokens=top_n,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                attempts += 1
                if attempts >= self.max_retries:
                    break
                delay = self.retry_base_delay * (2 ** (attempts - 1))
                status_code = _extract_status_code(exc)
                if status_code == 429:
                    delay = max(delay, self.rate_limit_delay)
                if delay > 0:
                    self._sleep(delay)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Failed to call Hugging Face Inference API")

    def _parse_response(self, response: object, top_k: int) -> List[Tuple[str, float]]:
        details = _get_attr(response, "details")
        token_entries = []
        if details is not None:
            raw_tokens = _get_attr(details, "tokens")
            if isinstance(raw_tokens, Sequence):
                token_entries = list(raw_tokens)

        top_tokens: List[Tuple[str, float]] = []
        if token_entries:
            first_entry = token_entries[0]
            raw_top_tokens = _get_attr(first_entry, "top_tokens") or []
            normalized = filter(None, (_normalize_token_entry(item) for item in raw_top_tokens))
            top_tokens = _unique_preserve_order(normalized)
            if not top_tokens:
                fallback = _normalize_token_entry(first_entry)
                if fallback is not None:
                    top_tokens = [fallback]
        if not top_tokens:
            generated = _get_attr(response, "generated_text")
            if isinstance(generated, str) and generated.strip():
                top_tokens = [(generated.strip(), 0.0)]

        return top_tokens[: int(top_k)]


def load_hf_settings_from_env() -> Dict[str, object]:
    """Return Hugging Face configuration derived from environment variables."""

    model_id = os.getenv("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
    api_token = os.getenv("HF_API_TOKEN") or None
    top_n_raw = os.getenv("HF_TOP_N_TOKENS", "10")
    temp_raw = os.getenv("ORACLE_TEMP", "0")
    expose_raw = os.getenv("ORACLE_EXPOSE_PROBS", "false")

    try:
        top_n_tokens = int(top_n_raw)
    except ValueError as exc:
        raise RuntimeError("HF_TOP_N_TOKENS must be an integer") from exc

    try:
        temperature = float(temp_raw)
    except ValueError as exc:
        raise RuntimeError("ORACLE_TEMP must be a number") from exc

    expose_probs = expose_raw.lower() in {"1", "true", "yes", "on"}

    return {
        "model_id": model_id,
        "api_token": api_token,
        "top_n_tokens": max(1, top_n_tokens),
        "temperature": temperature,
        "expose_probs": expose_probs,
    }


def build_hf_client_from_env() -> HuggingFaceServerlessProvider:
    """Helper to build a provider from environment variables."""

    settings = load_hf_settings_from_env()
    return HuggingFaceServerlessProvider(**settings)

