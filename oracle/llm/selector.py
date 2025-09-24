"""Select appropriate LLM provider based on environment variables."""

from __future__ import annotations

import os

from .base import SequenceProvider
from .hf_serverless import build_hf_client_from_env
from .llama_cpp_local import LlamaCppLocalProvider
from .transformers_local import TransformersLocalProvider

DEFAULT_TOP_K = 5


def _parse_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _ensure_local_enabled() -> None:
    use_local = _parse_bool_env("ORACLE_USE_LOCAL", True)
    if not use_local:
        raise RuntimeError(
            "Local LLM usage is required but ORACLE_USE_LOCAL is not enabled"
        )


def _build_local_provider() -> SequenceProvider:
    _ensure_local_enabled()
    backend = os.getenv("ORACLE_LLM_BACKEND")
    if not backend:
        raise RuntimeError(
            "ORACLE_LLM_BACKEND must be set to 'transformers' or 'llama_cpp'"
        )

    backend = backend.lower()
    if backend == "transformers":
        model_id = os.getenv("ORACLE_MODEL_ID")
        if not model_id:
            raise RuntimeError("ORACLE_MODEL_ID must be set for transformers backend")
        return TransformersLocalProvider(model_id=model_id)
    if backend == "llama_cpp":
        model_path = os.getenv("ORACLE_GGUF_PATH")
        if not model_path:
            raise RuntimeError("ORACLE_GGUF_PATH must be set for llama_cpp backend")
        return LlamaCppLocalProvider(model_path=model_path)

    raise RuntimeError(f"Unsupported ORACLE_LLM_BACKEND: {backend}")


def build_sequence_provider() -> SequenceProvider:
    provider = os.getenv("ORACLE_LLM_PROVIDER")
    if provider:
        provider = provider.lower()
        if provider == "hf_serverless":
            return build_hf_client_from_env()
        raise RuntimeError(f"Unsupported ORACLE_LLM_PROVIDER: {provider}")

    return _build_local_provider()


def get_top_k(default: int = DEFAULT_TOP_K) -> int:
    value = os.getenv("ORACLE_TOP_K")
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise RuntimeError("ORACLE_TOP_K must be an integer") from exc
    return max(1, parsed)

