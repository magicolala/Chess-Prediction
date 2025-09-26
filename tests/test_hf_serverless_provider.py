import types

import pytest

from oracle.llm.hf_serverless import (
    SAFE_SERVERLESS_MODELS,
    HuggingFaceServerlessProvider,
    _ServerlessCandidate,
    _extract_status_code,
    DEFAULT_SERVERLESS_PROVIDER,
)


class _Dummy404Error(Exception):
    status_code = 404


class _Always404Client:
    def __init__(self):
        self.calls = 0

    def text_generation(self, *args, **kwargs):
        self.calls += 1
        raise _Dummy404Error("missing model")


class _Dummy403Error(Exception):
    status_code = 403


class _Always403Client:
    def __init__(self):
        self.calls = 0

    def text_generation(self, *args, **kwargs):
        self.calls += 1
        raise _Dummy403Error("forbidden")


def _build_provider_for_test(models, client_cls=_Always404Client):
    provider = object.__new__(HuggingFaceServerlessProvider)
    provider.models = list(models)
    provider._model_idx = 0
    provider._client_cache = {model: client_cls() for model in provider.models}
    provider._client_factory = None
    provider.client = provider._client_cache[provider.models[0]]
    provider._model_providers = {}
    provider._has_api_token = True
    provider.max_retries = 1
    provider.retry_base_delay = 0.0
    provider.rate_limit_delay = 0.0
    provider._sleep = lambda _: None
    provider.do_sample = False
    provider.temperature = 0.0
    return provider


def test_safe_serverless_models_default_list(monkeypatch):
    provider = object.__new__(HuggingFaceServerlessProvider)
    monkeypatch.delenv("HF_MODEL_CANDIDATES", raising=False)
    monkeypatch.setattr(
        "oracle.llm.hf_serverless._discover_serverless_models", lambda limit=25: []
    )
    candidates = provider._build_candidate_list("")
    assert candidates[0] == "meta-llama/Llama-3.1-8B-Instruct"
    expected_tail = [
        model
        for model in SAFE_SERVERLESS_MODELS
        if model != "meta-llama/Llama-3.1-8B-Instruct"
    ]
    assert candidates[1 : 1 + len(expected_tail)] == expected_tail


def test_candidate_list_includes_discovered_models(monkeypatch):
    provider = object.__new__(HuggingFaceServerlessProvider)
    monkeypatch.delenv("HF_MODEL_CANDIDATES", raising=False)
    monkeypatch.setattr(
        "oracle.llm.hf_serverless._discover_serverless_models",
        lambda limit=25: [
            _ServerlessCandidate("discovered-a", "hf-inference"),
            _ServerlessCandidate("discovered-b", "hf-inference"),
        ],
    )
    candidates = provider._build_candidate_list("primary-model")
    assert candidates[:3] == ["primary-model", "discovered-a", "discovered-b"]


def test_call_with_retries_reraises_with_hint():
    provider = _build_provider_for_test(["missing-one", "missing-two"])
    with pytest.raises(RuntimeError) as excinfo:
        provider._call_with_retries("prompt", 3)
    message = str(excinfo.value)
    assert "returned 404" in message
    assert "missing-one" in message and "missing-two" in message


def test_call_with_retries_surfaces_auth_errors():
    provider = _build_provider_for_test(["needs-auth"], client_cls=_Always403Client)
    with pytest.raises(RuntimeError) as excinfo:
        provider._call_with_retries("prompt", 3)
    assert "authorization error" in str(excinfo.value)


@pytest.mark.parametrize(
    "error, expected",
    [
        (_Dummy404Error(), 404),
        (types.SimpleNamespace(status_code=503), 503),
        (types.SimpleNamespace(response=types.SimpleNamespace(status_code=401)), 401),
        (types.SimpleNamespace(request=types.SimpleNamespace(status_code=500)), 500),
        (ValueError("no code"), None),
    ],
)
def test_extract_status_code_variants(error, expected):
    assert _extract_status_code(error) == expected


def test_prune_unroutable_models_drops_known_bad_routes(monkeypatch):
    provider = object.__new__(HuggingFaceServerlessProvider)
    provider.provider = DEFAULT_SERVERLESS_PROVIDER
    provider.models = ["routable", "bad-model"]
    provider._model_idx = 0
    provider._model_providers = {"routable": DEFAULT_SERVERLESS_PROVIDER}

    def fake_resolve(self, model):  # noqa: ANN001
        if model == "bad-model":
            self._model_providers[model] = None
        return self._model_providers.get(model)

    monkeypatch.setattr(
        HuggingFaceServerlessProvider,
        "_resolve_provider_for_model",
        fake_resolve,
        raising=False,
    )

    provider._prune_unroutable_models()
    assert provider.models == ["routable"]


def test_prune_unroutable_models_keeps_unknown_when_lookup_unavailable(monkeypatch):
    provider = object.__new__(HuggingFaceServerlessProvider)
    provider.provider = DEFAULT_SERVERLESS_PROVIDER
    provider.models = ["unknown", "cached"]
    provider._model_idx = 0
    provider._model_providers = {"cached": DEFAULT_SERVERLESS_PROVIDER}

    def fake_resolve(self, model):  # noqa: ANN001
        if model == "unknown":
            return None
        return self._model_providers.get(model)

    monkeypatch.setattr(
        HuggingFaceServerlessProvider,
        "_resolve_provider_for_model",
        fake_resolve,
        raising=False,
    )

    provider._prune_unroutable_models()
    assert provider.models == ["unknown", "cached"]


def test_prune_unroutable_models_errors_when_all_removed(monkeypatch):
    provider = object.__new__(HuggingFaceServerlessProvider)
    provider.provider = DEFAULT_SERVERLESS_PROVIDER
    provider.models = ["bad"]
    provider._model_idx = 0
    provider._model_providers = {}

    def fake_resolve(self, model):  # noqa: ANN001
        self._model_providers[model] = None
        return None

    monkeypatch.setattr(
        HuggingFaceServerlessProvider,
        "_resolve_provider_for_model",
        fake_resolve,
        raising=False,
    )

    with pytest.raises(RuntimeError):
        provider._prune_unroutable_models()
