import pytest

from oracle.llm import selector
from oracle.llm.hf_serverless import HuggingFaceServerlessProvider


class DummyTransformersProvider:
    def __init__(self, model_id: str):
        self.model_id = model_id


class DummyLlamaProvider:
    def __init__(self, model_path: str):
        self.model_path = model_path


def test_selector_returns_transformers_provider(monkeypatch):
    monkeypatch.setenv("ORACLE_USE_LOCAL", "1")
    monkeypatch.setenv("ORACLE_LLM_BACKEND", "transformers")
    monkeypatch.setenv("ORACLE_MODEL_ID", "dummy-model")

    monkeypatch.setattr(
        selector, "TransformersLocalProvider", DummyTransformersProvider
    )

    provider = selector.build_sequence_provider()
    assert isinstance(provider, DummyTransformersProvider)
    assert provider.model_id == "dummy-model"


def test_selector_returns_llama_provider(monkeypatch):
    monkeypatch.setenv("ORACLE_USE_LOCAL", "1")
    monkeypatch.setenv("ORACLE_LLM_BACKEND", "llama_cpp")
    monkeypatch.setenv("ORACLE_GGUF_PATH", "/tmp/fake.gguf")

    monkeypatch.setattr(selector, "LlamaCppLocalProvider", DummyLlamaProvider)

    provider = selector.build_sequence_provider()
    assert isinstance(provider, DummyLlamaProvider)
    assert provider.model_path == "/tmp/fake.gguf"


def test_selector_requires_backend(monkeypatch):
    monkeypatch.delenv("ORACLE_LLM_BACKEND", raising=False)
    monkeypatch.setenv("ORACLE_USE_LOCAL", "1")

    with pytest.raises(RuntimeError):
        selector.build_sequence_provider()


def test_selector_requires_model_id(monkeypatch):
    monkeypatch.setenv("ORACLE_USE_LOCAL", "1")
    monkeypatch.setenv("ORACLE_LLM_BACKEND", "transformers")
    monkeypatch.delenv("ORACLE_MODEL_ID", raising=False)

    with pytest.raises(RuntimeError):
        selector.build_sequence_provider()


def test_selector_unknown_backend(monkeypatch):
    monkeypatch.setenv("ORACLE_USE_LOCAL", "1")
    monkeypatch.setenv("ORACLE_LLM_BACKEND", "unknown")

    with pytest.raises(RuntimeError):
        selector.build_sequence_provider()


def test_selector_hf_serverless(monkeypatch):
    monkeypatch.delenv("ORACLE_USE_LOCAL", raising=False)
    monkeypatch.setenv("ORACLE_LLM_PROVIDER", "hf_serverless")
    monkeypatch.setenv("HF_MODEL_ID", "test/model")
    monkeypatch.setenv("HF_TOP_N_TOKENS", "7")
    monkeypatch.setenv("HF_TEMPERATURE", "0")

    provider = selector.build_sequence_provider()

    assert isinstance(provider, HuggingFaceServerlessProvider)
    assert provider.model_id == "test/model"
    assert provider.top_n_tokens == 7
