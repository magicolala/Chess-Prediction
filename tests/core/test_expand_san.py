import math

from oracle.core.expand import expand_san_candidates


class FakeProvider:
    def __init__(self, mapping):
        self.mapping = mapping

    def get_top_sequences(self, prompt, legal_moves, depth, prob_threshold, top_k):
        return self.mapping.get(prompt, [])


def test_expand_san_candidates_accumulates_logprobs():
    mapping = {
        "": [("e", math.log(0.6)), ("d", math.log(0.4))],
        "e": [("4", math.log(0.7)), ("5", math.log(0.3))],
        "d": [("4", math.log(1.0))],
    }
    provider = FakeProvider(mapping)
    legal_moves = {"e4", "e5", "d4"}

    results = expand_san_candidates(
        provider=provider,
        prompt="",
        legal_moves=legal_moves,
        depth=2,
        prob_threshold=0.0,
        top_k=3,
    )

    probs = {move: math.exp(logprob) for move, logprob in results}
    assert math.isclose(probs["e4"], 0.42, rel_tol=1e-6)
    assert math.isclose(probs["e5"], 0.18, rel_tol=1e-6)
    assert math.isclose(probs["d4"], 0.4, rel_tol=1e-6)


def test_expand_san_prunes_non_matching_prefixes():
    mapping = {
        "": [("N", math.log(0.5))],
        "N": [("f", math.log(0.5)), ("x", math.log(0.5))],
        "Nf": [("3", math.log(1.0))],
        "Nx": [("e", math.log(1.0))],
    }
    provider = FakeProvider(mapping)
    legal_moves = {"Nf3"}

    results = expand_san_candidates(
        provider=provider,
        prompt="",
        legal_moves=legal_moves,
        depth=3,
        prob_threshold=0.0,
        top_k=3,
    )

    assert len(results) == 1
    move, logprob = results[0]
    assert move == "Nf3"
    assert math.isclose(math.exp(logprob), 0.25, rel_tol=1e-6)
