import math

from oracle.core.castling import split_castling


def test_split_castling_distributes_mass():
    base_logprob = math.log(0.5)
    continuation_logprob = math.log(0.2)

    results = split_castling("O-O", base_logprob, continuation_logprob)

    assert {move for move, _ in results} == {"O-O", "O-O-O"}

    probs = {move: math.exp(lp) for move, lp in results}
    assert math.isclose(probs["O-O"], 0.4, rel_tol=1e-6)
    assert math.isclose(probs["O-O-O"], 0.1, rel_tol=1e-6)


def test_split_castling_returns_original_when_no_continuation():
    base_logprob = math.log(0.5)
    results = split_castling("O-O", base_logprob, None)

    assert results == [("O-O", base_logprob)]


def test_split_castling_passthrough_for_other_moves():
    base_logprob = math.log(0.3)
    results = split_castling("e4", base_logprob, math.log(0.2))

    assert results == [("e4", base_logprob)]
