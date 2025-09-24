"""LLM provider protocol definitions."""

from typing import List, Protocol, Tuple


class SequenceProvider(Protocol):
    """Protocol describing sequence generation providers."""

    def get_top_sequences(
        self,
        prompt: str,
        legal_moves: List[str],
        depth: int,
        prob_threshold: float,
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """Return top sequences with their log probabilities."""

