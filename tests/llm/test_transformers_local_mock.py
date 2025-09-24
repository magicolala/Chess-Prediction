import math
from types import SimpleNamespace

import pytest

from oracle.llm.transformers_local import TransformersLocalProvider


class DummyTensor:
    def __init__(self, data):
        self._data = data

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._data


class DummyTokenizer:
    def __init__(self):
        self.decode_map = {
            0: "e4",
            1: "d4",
            2: "c4",
        }

    def __call__(self, prompt: str, return_tensors: str):
        assert prompt == "Prompt"
        assert return_tensors == "pt"
        return {"input_ids": [[1, 2, 3]]}

    def decode(self, token_ids, skip_special_tokens=True):
        assert skip_special_tokens
        (token_id,) = token_ids
        return self.decode_map[token_id]


class DummyModel:
    def __init__(self):
        self.seen_kwargs = None

    def generate(self, **kwargs):
        self.seen_kwargs = kwargs
        scores = DummyTensor([[2.0, 1.0, 0.0]])
        return SimpleNamespace(
            sequences=[[1, 2, 3, 0]],
            scores=[scores],
        )


class DummyTokenizerWrapper(DummyTokenizer):
    def convert_ids_to_tokens(self, token_id):
        return self.decode_map[token_id]


@pytest.fixture
def provider():
    dummy_tokenizer = DummyTokenizerWrapper()
    dummy_model = DummyModel()

    provider = TransformersLocalProvider(
        model_id="dummy",
        tokenizer=dummy_tokenizer,
        model=dummy_model,
    )

    return provider, dummy_model


def test_transformers_provider_returns_sorted_logprobs(provider):
    provider_instance, dummy_model = provider
    results = provider_instance.get_top_sequences(
        prompt="Prompt",
        legal_moves=["e4", "d4", "c4"],
        depth=1,
        prob_threshold=0.0,
        top_k=2,
    )

    assert len(results) == 2
    assert [move for move, _ in results] == ["e4", "d4"]

    exp_values = [math.exp(v) for v in [2.0, 1.0, 0.0]]
    total = sum(exp_values)
    expected_logprobs = [math.log(exp_values[i] / total) for i in range(2)]
    for (_, logprob), expected in zip(results, expected_logprobs):
        assert pytest.approx(logprob, rel=1e-6) == expected

    assert dummy_model.seen_kwargs["do_sample"] is False
    assert dummy_model.seen_kwargs["temperature"] == 0
    assert dummy_model.seen_kwargs["max_new_tokens"] == 1
    assert dummy_model.seen_kwargs["output_scores"] is True
    assert dummy_model.seen_kwargs["return_dict_in_generate"] is True
