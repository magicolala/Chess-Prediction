import types

import pytest

from oracle.llm.hf_serverless import (
    SAFE_SERVERLESS_MODELS,
    HuggingFaceServerlessProvider,
    _extract_status_code,
)


class _Dummy404Error(Exception):
    status_code = 404


class _Always404Client:
    def __init__(self):
        self.calls = 0

    def text_generation(self, *args, **kwargs):
        self.calls += 1
        raise _Dummy404Error("missing model")


def _build_provider_for_test(models):
    provider = object.__new__(HuggingFaceServerlessProvider)
    provider.models = list(models)
    provider._model_idx = 0
    provider._client_cache = {model: _Always404Client() for model in provider.models}
    provider._client_factory = None
    provider.client = provider._client_cache[provider.models[0]]
    provider.max_retries = 1
    provider.retry_base_delay = 0.0
    provider.rate_limit_delay = 0.0
    provider._sleep = lambda _: None
    provider.do_sample = False
    provider.temperature = 0.0
    return provider


def test_safe_serverless_models_default_list():
    provider = object.__new__(HuggingFaceServerlessProvider)
    candidates = provider._build_candidate_list("")
    assert candidates[0] == SAFE_SERVERLESS_MODELS[0]
    assert candidates[: len(SAFE_SERVERLESS_MODELS)] == SAFE_SERVERLESS_MODELS


def test_call_with_retries_reraises_with_hint():
    provider = _build_provider_for_test(["missing-one", "missing-two"])
    with pytest.raises(RuntimeError) as excinfo:
        provider._call_with_retries("prompt", 3)
    message = str(excinfo.value)
    assert "returned 404" in message
    assert "missing-one" in message and "missing-two" in message


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
