from types import SimpleNamespace

from oracle.llm.hf_serverless import HuggingFaceServerlessProvider
from oracle.llm.hf_serverless import load_hf_settings_from_env


class DummyClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def text_generation(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response


FAKE_RESPONSE = SimpleNamespace(
    generated_text="e4",
    details=SimpleNamespace(
        tokens=[
            SimpleNamespace(
                id=42,
                text="e",
                logprob=-0.05,
                top_tokens=[
                    SimpleNamespace(token="e", logprob=-0.05),
                    SimpleNamespace(token="d", logprob=-0.25),
                ],
            )
        ],
        prefill=[SimpleNamespace(id=1, text="...")],
    ),
)


def build_provider(response=FAKE_RESPONSE, **kwargs):
    client = DummyClient(response)
    provider = HuggingFaceServerlessProvider(
        model_id="mistral", client=client, top_n_tokens=5, **kwargs
    )
    return provider, client


def test_hf_serverless_defaults_provider(monkeypatch):
    captured = {}

    class FakeInferenceClient:  # noqa: D401 - simple stub for dependency injection
        """Capture initialization kwargs from HuggingFaceServerlessProvider."""

        def __init__(self, *, model, token=None, provider=None, **kwargs):
            captured["model"] = model
            captured["token"] = token
            captured["provider"] = provider

        def text_generation(self, *args, **kwargs):  # pragma: no cover - not exercised
            raise AssertionError("text_generation should not be called")

    monkeypatch.setattr(
        "oracle.llm.hf_serverless.InferenceClient", FakeInferenceClient
    )

    provider = HuggingFaceServerlessProvider(model_id="mixtral", api_token="tok")

    assert provider.provider == "hf-inference"
    assert captured == {"model": "mixtral", "token": "tok", "provider": "hf-inference"}


def test_hf_serverless_provider_env_override(monkeypatch):
    monkeypatch.setenv("HF_PROVIDER", "auto")
    settings = load_hf_settings_from_env()
    assert settings["provider"] == "auto"

    captured = {}

    class FakeInferenceClient:
        def __init__(self, *, model, token=None, provider=None, **kwargs):
            captured["provider"] = provider

        def text_generation(self, *args, **kwargs):  # pragma: no cover - not exercised
            raise AssertionError("text_generation should not be called")

    monkeypatch.setattr(
        "oracle.llm.hf_serverless.InferenceClient", FakeInferenceClient
    )

    provider = HuggingFaceServerlessProvider(**settings)
    assert provider.provider == "auto"
    assert captured["provider"] == "auto"


def test_hf_serverless_returns_top_sequences():
    provider, client = build_provider()

    results = provider.get_top_sequences("Prompt", ["e4"], depth=1, prob_threshold=0.0, top_k=5)

    assert results == [("e", -0.05), ("d", -0.25)]

    assert len(client.calls) == 1
    args, kwargs = client.calls[0]
    assert args == ("Prompt",)
    assert kwargs["max_new_tokens"] == 1
    assert kwargs["details"] is True
    assert kwargs["return_full_text"] is False
    assert kwargs["do_sample"] is False
    assert kwargs["temperature"] == 0.0
    assert kwargs["top_n_tokens"] == 5


def test_hf_serverless_fallback_to_generated_token():
    response = SimpleNamespace(
        generated_text=" d4",
        details=SimpleNamespace(
            tokens=[SimpleNamespace(id=1, text=" d", logprob=-0.1, top_tokens=[])],
            prefill=[],
        ),
    )
    provider, _ = build_provider(response=response)

    results = provider.get_top_sequences("Prompt", ["d4"], depth=1, prob_threshold=0.0, top_k=3)

    assert results == [("d", -0.1)]


def test_hf_serverless_respects_top_k_limit():
    response = SimpleNamespace(
        generated_text="e4",
        details=SimpleNamespace(
            tokens=[
                SimpleNamespace(
                    id=42,
                    text="e",
                    logprob=-0.05,
                    top_tokens=[
                        SimpleNamespace(token="e", logprob=-0.05),
                        SimpleNamespace(token="d", logprob=-0.25),
                        SimpleNamespace(token="c", logprob=-1.2),
                    ],
                )
            ],
            prefill=[],
        ),
    )
    provider, _ = build_provider(response=response)

    results = provider.get_top_sequences("Prompt", ["e4"], depth=1, prob_threshold=0.0, top_k=2)

    assert results == [("e", -0.05), ("d", -0.25)]
