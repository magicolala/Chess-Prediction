"""Hugging Face Inference API-backed sequence provider."""

from __future__ import annotations

import math
import os
import time
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency import
    from huggingface_hub import HfApi, list_models
except Exception:  # pragma: no cover - handled lazily
    list_models = None  # type: ignore[misc]
    HfApi = None  # type: ignore[misc]

from .base import SequenceProvider

try:  # pragma: no cover - optional dependency import
    from huggingface_hub import InferenceClient
except Exception:  # pragma: no cover - handled lazily
    InferenceClient = None  # type: ignore[misc]

# NOTE: ``SAFE_SERVERLESS_CANDIDATES`` provides a static safety net when runtime
# discovery fails (e.g. because `huggingface_hub` is not installed in the user's
# environment). The entries are restricted to checkpoints currently routed via
# the ``hf-inference`` provider as of 2025-09-25 so that Oracle never attempts
# providers requiring third-party API keys without an explicit override.
DEFAULT_SERVERLESS_PROVIDER = "hf-inference"


class _ServerlessCandidate(NamedTuple):
    model_id: str
    provider: Optional[str] = None


SAFE_SERVERLESS_CANDIDATES: List[_ServerlessCandidate] = [
    _ServerlessCandidate("meta-llama/Llama-3.1-8B-Instruct"),
    _ServerlessCandidate("mistralai/Mistral-7B-Instruct-v0.3"),
    _ServerlessCandidate("google/gemma-2-2b-it"),
]

SAFE_SERVERLESS_MODELS = [candidate.model_id for candidate in SAFE_SERVERLESS_CANDIDATES]

_MISSING_DEPENDENCY_MSG = (
    "huggingface-hub must be installed to use HuggingFaceServerlessProvider"
)


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


def _unique_preserve_order(
    pairs: Iterable[Tuple[str, float]],
) -> List[Tuple[str, float]]:
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
        provider: str | None = None,
        max_retries: int = 3,
        retry_base_delay: float = 0.5,
        rate_limit_delay: float = 1.0,
        sleep: Optional[Callable[[float], None]] = None,
    ) -> None:
        self.api_token = (api_token or "").strip()
        self._has_api_token = bool(self.api_token)
        self.top_n_tokens = max(1, int(top_n_tokens or 1))
        self.temperature = float(temperature)
        self.do_sample = (
            bool(do_sample) if do_sample is not None else self.temperature > 0
        )
        self.expose_probs = bool(expose_probs) if expose_probs is not None else False
        self.provider = (provider or "hf-inference").strip() or "hf-inference"
        self.max_retries = max(1, int(max_retries or 1))
        self.retry_base_delay = max(0.0, float(retry_base_delay))
        self.rate_limit_delay = max(0.0, float(rate_limit_delay))
        self._sleep = sleep if sleep is not None else time.sleep

        self._client_factory = None
        self._client_cache: Dict[str, object] = {}
        self._model_providers: Dict[str, Optional[str]] = {}
        self.models = self._build_candidate_list(model_id)
        self._model_idx = 0
        self._prune_inaccessible_models()
        if not self.models:
            raise RuntimeError(
                "No Hugging Face serverless models are available. Accept the model's "
                "usage terms in your Hugging Face account or set HF_MODEL_ID / "
                "HF_MODEL_CANDIDATES to checkpoints you can access."
            )
        first_model = self.current_model

        if client is not None:
            self.models = [first_model]
            self.client = client
            self._client_cache[first_model] = client
            self._has_api_token = True
        else:
            if InferenceClient is None:  # pragma: no cover - dependency guard
                raise ImportError(_MISSING_DEPENDENCY_MSG)
            self._client_factory = self._create_client
            if not self._has_api_token:
                raise RuntimeError(
                    "No Hugging Face API token configured. Set HF_API_TOKEN or "
                    "HUGGINGFACEHUB_API_TOKEN before using the serverless provider.",
                )
            self.client = self._get_client_for_model(self.current_model)

        self.model_id = self.current_model

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
        last_exc: Optional[Exception] = None
        if not self._has_api_token and self._client_factory is not None:
            raise RuntimeError(
                "No Hugging Face API token configured. Provide a valid token via the "
                "HF_API_TOKEN environment variable or the web UI before requesting "
                "serverless analysis.",
            )

        for idx, model in enumerate(self.models):
            client = self._get_client_for_model(model)
            self._model_idx = idx
            self.client = client
            self.model_id = model
            attempts = 0
            failed_for_model = False
            while attempts < self.max_retries:
                try:
                    return client.text_generation(
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
                    failed_for_model = True
                    attempts += 1
                    status_code = _extract_status_code(exc)
                    if status_code == 404:
                        print(
                            f"HF serverless 404 on {model}, skipping candidate...",
                            flush=True,
                        )
                        break
                    if self._should_switch_model(exc):
                        break
                    if attempts >= self.max_retries:
                        break
                    delay = self.retry_base_delay * (2 ** (attempts - 1))
                    if status_code == 429:
                        delay = max(delay, self.rate_limit_delay)
                    if delay > 0:
                        self._sleep(delay)
            if failed_for_model and idx + 1 < len(self.models):
                print(
                    f"HF serverless fallback -> {self.models[idx + 1]}",
                    flush=True,
                )
        if last_exc is not None:
            status = _extract_status_code(last_exc)
            if status in {401, 403}:
                raise RuntimeError(
                    "Hugging Face Inference API rejected the request with an "
                    "authorization error. Provide an HF_API_TOKEN that has "
                    "access to the selected checkpoints (and accept any "
                    "required model terms) before retrying."
                ) from last_exc
            if status == 404:
                candidate_details = []
                for candidate in self.models:
                    provider_hint = self._resolve_provider_for_model(candidate)
                    if provider_hint and provider_hint != self.provider:
                        candidate_details.append(f"{candidate} (provider {provider_hint})")
                    else:
                        candidate_details.append(candidate)
                candidate_list = ", ".join(candidate_details)
                raise RuntimeError(
                    "All configured Hugging Face serverless models returned 404 (not found). "
                    "This usually means your Hugging Face account lacks access to the "
                    "selected provider or you have not accepted the model's usage terms. "
                    "Verify that your HF_MODEL_ID or HF_MODEL_CANDIDATES only reference "
                    "available serverless checkpoints and ensure your HF_API_TOKEN has "
                    "permissions for their backing provider. Tried: "
                    f"{candidate_list or 'none'}."
                ) from last_exc
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
            normalized = filter(
                None, (_normalize_token_entry(item) for item in raw_top_tokens)
            )
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

    # Candidate management ----------------------------------------------------
    def _build_candidate_list(self, model_id: str) -> List[str]:
        if not hasattr(self, "_model_providers"):
            self._model_providers = {}
        provider_map: Dict[str, Optional[str]] = {}
        env_value = os.getenv("HF_MODEL_CANDIDATES", "")
        if env_value:
            raw_candidates = [
                _parse_candidate_entry(item)
                for item in env_value.split(",")
                if item.strip()
            ]
        else:
            primary = model_id or "meta-llama/Llama-3.1-8B-Instruct"
            discovered = _discover_serverless_models()
            raw_candidates = [
                _ServerlessCandidate(primary),
                *discovered,
                *SAFE_SERVERLESS_CANDIDATES,
            ]
        if model_id and all(entry.model_id != model_id for entry in raw_candidates):
            raw_candidates.insert(0, _ServerlessCandidate(model_id))
        candidates: List[str] = []
        seen = set()
        for candidate in raw_candidates:
            model_name = candidate.model_id.strip()
            if not model_name or model_name in seen:
                continue
            candidates.append(model_name)
            seen.add(model_name)
            if candidate.provider:
                provider_map[model_name] = candidate.provider
        if provider_map:
            self._model_providers.update(provider_map)
        if not candidates:
            raise RuntimeError("No Hugging Face models configured")
        return candidates

    def _prune_inaccessible_models(self) -> None:
        """Remove candidates that lack a routable serverless provider."""

        if not self.models:
            return
        # Respect explicit provider overrides such as "auto" or custom endpoints.
        if self.provider and self.provider not in {DEFAULT_SERVERLESS_PROVIDER, ""}:
            return
        if HfApi is None:
            return
        try:
            api = HfApi()
        except Exception:  # pragma: no cover - dependency/environment errors
            return

        filtered: List[str] = []
        for model in self.models:
            provider_override = self._model_providers.get(model)
            resolved = self._lookup_inference_provider(api, model, provider_override)
            if resolved is None:
                print(
                    "HF serverless skipping "
                    f"{model} (no {provider_override or DEFAULT_SERVERLESS_PROVIDER} route)",
                    flush=True,
                )
                continue
            if provider_override is None and resolved:
                self._model_providers[model] = resolved
            filtered.append(model)

        if filtered:
            self.models = filtered
            self._model_idx = min(self._model_idx, len(self.models) - 1)
        else:
            self.models = []

    def _lookup_inference_provider(
        self,
        api: "HfApi",  # type: ignore[name-defined]
        model: str,
        explicit_provider: Optional[str],
    ) -> Optional[str]:
        """Return a provider capable of serving text generation for ``model``."""

        try:
            info = api.model_info(model, expand=["inferenceProviderMapping"])
        except Exception:  # pragma: no cover - network/auth failures
            return explicit_provider or DEFAULT_SERVERLESS_PROVIDER

        mapping = getattr(info, "inference_provider_mapping", None) or []
        preferred: Optional[str] = None
        fallback: Optional[str] = None
        for entry in mapping:
            provider_name = getattr(entry, "provider", None)
            task = getattr(entry, "task", "") or ""
            if not provider_name or task not in {"text-generation", "conversational"}:
                continue
            if explicit_provider:
                if provider_name == explicit_provider:
                    preferred = provider_name
                    break
                continue
            if provider_name == DEFAULT_SERVERLESS_PROVIDER:
                preferred = provider_name
                break
            if fallback is None:
                fallback = provider_name
        return preferred or fallback or explicit_provider

    @property
    def current_model(self) -> str:
        return self.models[self._model_idx]

    def _create_client(self, model: str):  # noqa: ANN101
        if InferenceClient is None:  # pragma: no cover - dependency guard
            raise ImportError(_MISSING_DEPENDENCY_MSG)
        provider = self._resolve_provider_for_model(model) or self.provider
        return InferenceClient(
            model=model,
            token=self.api_token or None,
            provider=provider,
        )

    def _get_client_for_model(self, model: str):  # noqa: ANN101
        if model in self._client_cache:
            return self._client_cache[model]
        if self._client_factory is None:
            return self.client
        client = self._client_factory(model)
        self._client_cache[model] = client
        return client

    @staticmethod
    def _should_switch_model(exc: Exception) -> bool:
        status_code = _extract_status_code(exc)
        if status_code == 404:
            return True
        message = str(exc).lower()
        if (
            "not supported for task text-generation" in message
            and "conversational" in message
        ):
            return True
        return "model" in message and "not found" in message

    def _resolve_provider_for_model(self, model: str) -> Optional[str]:
        provider = self._model_providers.get(model)
        if provider is not None:
            return provider
        # Respect explicit provider overrides such as "auto" or custom endpoints.
        provider_override = getattr(self, "provider", None)
        if provider_override and provider_override != DEFAULT_SERVERLESS_PROVIDER:
            return provider_override
        if HfApi is None:
            return None
        try:
            api = HfApi()
            info = api.model_info(model, expand=["inferenceProviderMapping"])
        except Exception:  # pragma: no cover - network/auth failures
            return None

        mapping = getattr(info, "inference_provider_mapping", None) or []
        preferred = None
        for entry in mapping:
            task = getattr(entry, "task", "") or ""
            provider_name = getattr(entry, "provider", None)
            if not provider_name:
                continue
            if provider_name == self.provider and task in {
                "text-generation",
                "conversational",
            }:
                preferred = provider_name
                break
            if preferred is None and task in {"text-generation", "conversational"}:
                preferred = provider_name
        if preferred:
            self._model_providers[model] = preferred
        return preferred


def load_hf_settings_from_env() -> Dict[str, object]:
    """Return Hugging Face configuration derived from environment variables."""

    model_id = os.getenv("HF_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
    api_token = (
        os.getenv("HF_API_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or None
    )
    top_n_raw = os.getenv("HF_TOP_N_TOKENS", "10")
    temp_raw = os.getenv("HF_TEMPERATURE", "0")
    expose_raw = os.getenv("ORACLE_EXPOSE_PROBS", "false")
    provider = os.getenv("HF_PROVIDER")

    try:
        top_n_tokens = int(top_n_raw)
    except ValueError as exc:
        raise RuntimeError("HF_TOP_N_TOKENS must be an integer") from exc

    try:
        temperature = float(temp_raw)
    except ValueError as exc:
        raise RuntimeError("HF_TEMPERATURE must be a number") from exc

    expose_probs = expose_raw.lower() in {"1", "true", "yes", "on"}

    return {
        "model_id": model_id,
        "api_token": api_token,
        "top_n_tokens": max(1, top_n_tokens),
        "temperature": temperature,
        "expose_probs": expose_probs,
        "provider": provider,
    }


def build_hf_client_from_env() -> HuggingFaceServerlessProvider:
    """Helper to build a provider from environment variables."""

    settings = load_hf_settings_from_env()
    provider = HuggingFaceServerlessProvider(**settings)
    models = getattr(provider, "models", None)
    if isinstance(models, Sequence) and models:
        candidate_list = ", ".join(str(model) for model in models)
        print(f"HF serverless model candidates: {candidate_list}")
    active_model = getattr(provider, "current_model", None) or provider.model_id
    print(f"HF serverless active model: {active_model}")
    return provider
def _parse_candidate_entry(raw: str) -> _ServerlessCandidate:
    """Parse an ``HF_MODEL_CANDIDATES`` entry into a candidate tuple."""

    text = raw.strip()
    if not text:
        return _ServerlessCandidate("")
    if "::" in text:
        provider, model = text.split("::", 1)
        return _ServerlessCandidate(model.strip(), provider.strip() or None)
    return _ServerlessCandidate(text)


_DISCOVERED_PROVIDERS: Dict[str, Optional[str]] = {}


def _discover_serverless_models(limit: int = 25) -> List[_ServerlessCandidate]:
    """Return warm Hugging Face serverless checkpoints ordered by size."""

    if list_models is None:
        return []

    try:
        candidates = list(
            list_models(
                inference="warm", pipeline_tag="text-generation", limit=limit
            )
        )
    except Exception:  # pragma: no cover - network/permission errors
        return []

    api: Optional[HfApi]
    if HfApi is None:
        api = None
    else:
        try:
            api = HfApi()
        except Exception:  # pragma: no cover - dependency/environment issues
            api = None

    ordered: List[_ServerlessCandidate] = []
    seen = set()
    for info in candidates:
        model_id = getattr(info, "modelId", None) or getattr(info, "id", None)
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        model_id = model_id.strip()
        lowered = model_id.lower()
        if "text-generation" not in lowered and "instruct" not in lowered:
            # Skip obviously unrelated checkpoints when discovery returns
            # heterogeneous tasks.
            if not any(keyword in lowered for keyword in ("chat", "zephyr")):
                continue
        if model_id in seen:
            continue
        provider = None
        if api is not None:
            try:
                info_full = api.model_info(
                    model_id, expand=["inferenceProviderMapping"]
                )
            except Exception:  # pragma: no cover - transient API errors
                info_full = None
            if info_full is not None:
                mapping = getattr(info_full, "inference_provider_mapping", None) or []
                for entry in mapping:
                    entry_provider = getattr(entry, "provider", None)
                    task = getattr(entry, "task", "") or ""
                    if (
                        entry_provider
                        and entry_provider == DEFAULT_SERVERLESS_PROVIDER
                        and task in {"text-generation", "conversational"}
                    ):
                        provider = entry_provider
                        break
        if provider != DEFAULT_SERVERLESS_PROVIDER:
            # Skip models that are not backed by the default provider because
            # they require custom API keys that Oracle cannot manage on behalf
            # of the user.
            continue
        ordered.append(_ServerlessCandidate(model_id, provider))
        seen.add(model_id)
        if provider:
            _DISCOVERED_PROVIDERS[model_id] = provider
    return ordered

