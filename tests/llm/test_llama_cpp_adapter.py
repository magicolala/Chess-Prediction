import math

import pytest

from oracle.llm.llama_cpp_local import LlamaCppLocalProvider


class FakeLlama:
    def __init__(self, model_path: str, logits_all: bool = False):
        assert logits_all
        self.model_path = model_path
        self.last_prompt = None
        self.last_tokens = None

    def tokenize(self, prompt: bytes, add_bos: bool = True):
        assert add_bos
        self.last_prompt = prompt
        return [101, 102]

    def eval(self, tokens, n_past=0):  # noqa: D401 - mimic llama signature
        self.last_tokens = tokens
        return None

    def get_logits(self):
        return [2.0, 1.0, 0.0]

    def detokenize(self, token_ids):
        mapping = {
            0: b"e4",
            1: b"d4",
            2: b"c4",
        }
        return mapping[token_ids[0]]


def test_llama_cpp_provider_returns_logprobs():
    fake_llama = FakeLlama("fake.gguf", logits_all=True)
    provider = LlamaCppLocalProvider(model_path="fake.gguf", llama=fake_llama)

    results = provider.get_top_sequences(
        prompt="1. e4",
        legal_moves=["e4", "d4", "c4"],
        depth=1,
        prob_threshold=0.0,
        top_k=2,
    )

    assert [move for move, _ in results] == ["e4", "d4"]

    exp_values = [math.exp(v) for v in [2.0, 1.0, 0.0]]
    total = sum(exp_values)
    expected_logprobs = [math.log(exp_values[i] / total) for i in range(2)]
    for (_, logprob), expected in zip(results, expected_logprobs):
        assert logprob == pytest.approx(expected, rel=1e-6)

    assert fake_llama.last_prompt == "1. e4".encode("utf-8")
    assert fake_llama.last_tokens == [101, 102]
