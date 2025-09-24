import io
import math

import chess
import chess.engine
import chess.pgn
import pytest

from oracle.calib.calibrator import Calibrator
from oracle.core.expected import expected_score_cp
from oracle.pipeline.analyze import analyze


class FakeProvider:
    def __init__(self, mapping):
        self.mapping = mapping

    def get_top_sequences(self, prompt, legal_moves, depth, prob_threshold, top_k):
        return self.mapping.get(prompt, [])


def fake_engine_factory(evaluations):
    class FakeEngine:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def analyse(self, board, limit):
            return {"score": evaluations[board.fen()]}

    def factory():
        return FakeEngine()

    return factory


def build_board(pgn: str):
    game = chess.pgn.read_game(io.StringIO(pgn))
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
    return board


def test_pipeline_with_fakes(monkeypatch):
    pgn = "1. e4 e5"
    board = build_board(pgn)

    mapping = {
        "": [("N", math.log(1.0))],
        "N": [("f", math.log(0.6)), ("c", math.log(0.4))],
        "Nf": [("3", math.log(1.0))],
        "Nc": [("3", math.log(1.0))],
    }
    provider = FakeProvider(mapping)

    nf3_board = board.copy()
    nf3_board.push_san("Nf3")
    nc3_board = board.copy()
    nc3_board.push_san("Nc3")

    evaluations = {
        nf3_board.fen(): chess.engine.PovScore(chess.engine.Cp(80), chess.WHITE),
        nc3_board.fen(): chess.engine.PovScore(chess.engine.Cp(10), chess.WHITE),
    }
    engine_factory = fake_engine_factory(evaluations)

    config = {
        "quality_bias": {"good": 0.2, "mistake": -0.4, "unknown": 0.0},
        "time_control_bias": {"rapid": 0.0},
        "elo": {"reference": 1500, "scale": 0.0},
    }
    calibrator = Calibrator(config=config)

    result = analyze(
        pgn=pgn,
        ctx={"elo": 1500, "time_control": "rapid"},
        provider=provider,
        engine_factory=engine_factory,
        calibrator=calibrator,
        depth=3,
        prob_threshold=0.0,
        top_k=2,
        prompt="",
    )

    moves = result["moves"]
    assert len(moves) == 2

    totals = sum(move["adjusted_pct"] for move in moves)
    assert pytest.approx(totals, abs=0.1) == 100.0

    move_map = {move["san"]: move for move in moves}
    assert move_map["Nf3"]["adjusted_pct"] > move_map["Nf3"]["prior_pct"]
    assert move_map["Nc3"]["adjusted_pct"] < move_map["Nc3"]["prior_pct"]

    expected_manual = (
        move_map["Nf3"]["adjusted_pct"] / 100.0 * expected_score_cp(80)
        + move_map["Nc3"]["adjusted_pct"] / 100.0 * expected_score_cp(10)
    )
    assert math.isclose(result["expected_score"], expected_manual, rel_tol=1e-6)

    assert result["usage"]["top_k"] == 2
