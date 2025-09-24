from typing import Any

import pytest

from oracle.llm.hf_serverless import HuggingFaceServerlessProvider


FAKE_RESP = {
    "generated_text": "e4",
    "details": {
        "prefill": [{"id": 1, "text": "..."}],
        "tokens": [
            {
                "id": 99,
                "text": "e",
                "logprob": -0.05,
                "top_tokens": [
                    {"token": "e", "logprob": -0.05},
                    {"token": "d", "logprob": -0.25},
                ],
            }
        ],
    },
}


class StubInferenceClient:
    def __init__(
        self,
        model: str,
        token: str | None = None,
        provider: str | None = None,
    ) -> None:
        self.model = model
        self.token = token
        self.provider = provider
        self.calls = 0

    def text_generation(self, *args: Any, **kwargs: Any):  # noqa: ANN401
        self.calls += 1
        behavior = BEHAVIORS[self.model]
        if isinstance(behavior, Exception):
            raise behavior
        return behavior


BEHAVIORS: dict[str, Any] = {}


@pytest.fixture(autouse=True)
def clear_behaviors():
    BEHAVIORS.clear()
    yield
    BEHAVIORS.clear()


def test_hf_serverless_switches_model_on_conversational_error(monkeypatch):
    monkeypatch.setenv("HF_MODEL_CANDIDATES", "model-a,model-b")

    created_models: list[str] = []

    def factory(model: str, token: str | None = None, provider: str | None = None):
        client = StubInferenceClient(model, token, provider)
        created_models.append(model)
        return client

    monkeypatch.setattr("oracle.llm.hf_serverless.InferenceClient", factory)

    BEHAVIORS.update(
        {
            "model-a": ValueError(
                "not supported for task text-generation and provider featherless-ai. Supported task: conversational."
            ),
            "model-b": FAKE_RESP,
        }
    )

    provider = HuggingFaceServerlessProvider(model_id="model-a", api_token="token")

    results = provider.get_top_sequences(
        "Prompt", ["e4"], depth=1, prob_threshold=0.0, top_k=5
    )

    assert created_models == [
        "model-a",
        "model-b",
    ], "Provider should try fallback model"
    assert results, "Fallback model should yield token distribution"
    token, logprob = results[0]
    assert isinstance(token, str)
    assert isinstance(logprob, float)


class FakeHttpError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def test_hf_serverless_switches_model_on_404(monkeypatch):
    monkeypatch.setenv("HF_MODEL_CANDIDATES", "model-x,model-y")

    created_models: list[str] = []

    def factory(model: str, token: str | None = None, provider: str | None = None):
        client = StubInferenceClient(model, token, provider)
        created_models.append(model)
        return client

    monkeypatch.setattr("oracle.llm.hf_serverless.InferenceClient", factory)

    BEHAVIORS.update(
        {
            "model-x": FakeHttpError(404, "Model model-x not found"),
            "model-y": FAKE_RESP,
        }
    )

    provider = HuggingFaceServerlessProvider(model_id="model-x", api_token="token")

    results = provider.get_top_sequences(
        "Prompt", ["e4"], depth=1, prob_threshold=0.0, top_k=5
    )

    assert created_models == ["model-x", "model-y"], "Should try fallback model on 404"
    assert results, "Fallback model should succeed after 404"


def test_default_candidates_skip_retired_models(monkeypatch):
    monkeypatch.delenv("HF_MODEL_CANDIDATES", raising=False)

    created_models: list[str] = []

    def factory(model: str, token: str | None = None, provider: str | None = None):
        client = StubInferenceClient(model, token, provider)
        created_models.append(model)
        return client

    monkeypatch.setattr("oracle.llm.hf_serverless.InferenceClient", factory)

    provider = HuggingFaceServerlessProvider(model_id="gpt2", api_token="token")

    assert (
        "tiiuae/falcon-7b-instruct" not in provider.models
    ), "Deprecated Falcon model should not be part of fallback list"
    # Ensure the provider still wires up the configured client
    assert created_models == [
        provider.current_model
    ], "Provider should create exactly one client during init"
