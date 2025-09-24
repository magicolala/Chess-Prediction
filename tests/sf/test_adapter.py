import chess
import chess.engine

from oracle.sf.adapter import evaluate_moves


class FakeEngine:
    def __init__(self, evaluations):
        self.evaluations = evaluations
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def analyse(self, board, limit):
        fen = board.fen()
        self.calls.append(fen)
        score = self.evaluations[fen]
        return {"score": score}

    def quit(self):  # pragma: no cover - compatibility
        pass


def fake_engine_factory(evaluations):
    def _factory():
        return FakeEngine(evaluations)

    return _factory


def test_evaluate_moves_returns_cp_scores():
    board = chess.Board()
    e4_board = board.copy()
    e4_board.push_san("e4")
    d4_board = board.copy()
    d4_board.push_san("d4")

    evaluations = {
        e4_board.fen(): chess.engine.PovScore(chess.engine.Cp(50), chess.WHITE),
        d4_board.fen(): chess.engine.PovScore(chess.engine.Cp(-20), chess.WHITE),
    }

    factory = fake_engine_factory(evaluations)

    results = evaluate_moves(board, ["e4", "d4"], factory, depth=10)

    assert results["e4"]["cp"] == 50
    assert results["d4"]["cp"] == -20
    assert results["e4"].get("mate") is None


def test_evaluate_moves_handles_mate_scores():
    board = chess.Board()
    mate_board = board.copy()
    mate_board.push_san("e4")

    evaluations = {
        mate_board.fen(): chess.engine.PovScore(chess.engine.Mate(2), chess.WHITE)
    }
    factory = fake_engine_factory(evaluations)

    results = evaluate_moves(board, ["e4"], factory, depth=10)

    assert results["e4"]["mate"] == 2
    assert "cp" not in results["e4"]
