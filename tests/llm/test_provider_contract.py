import math
from typing import List, Tuple

import pytest

from oracle.llm.base import SequenceProvider


class FakeProvider:
    def get_top_sequences(
        self,
        prompt: str,
        legal_moves: List[str],
        depth: int,
        prob_threshold: float,
        top_k: int,
    ) -> List[Tuple[str, float]]:
        return [("e4", math.log(0.9)), ("d4", math.log(0.1))]


def test_fake_provider_matches_protocol() -> None:
    provider: SequenceProvider
    provider = FakeProvider()

    results = provider.get_top_sequences("Prompt", ["e4", "d4"], 1, 0.05, 2)

    assert isinstance(results, list)
    assert all(isinstance(item, tuple) for item in results)
    assert all(len(item) == 2 for item in results)
    assert all(isinstance(item[0], str) for item in results)
    assert all(isinstance(item[1], float) for item in results)
