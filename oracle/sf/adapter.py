"""Adapter utilities for communicating with Stockfish engines."""

from __future__ import annotations

from typing import Callable, Dict, Iterable

import chess
import chess.engine


EngineFactory = Callable[[], chess.engine.SimpleEngine]


def evaluate_moves(
    board: chess.Board,
    san_moves: Iterable[str],
    engine_factory: Callable[[], chess.engine.SimpleEngine],
    depth: int = 12,
) -> Dict[str, Dict[str, int]]:
    """Evaluate SAN moves using the provided engine factory."""

    results: Dict[str, Dict[str, int]] = {}
    san_list = list(san_moves)
    if not san_list:
        return results

    limit = chess.engine.Limit(depth=depth)
    with engine_factory() as engine:
        for san in san_list:
            analysis_board = board.copy()
            try:
                move = analysis_board.parse_san(san)
            except ValueError:
                results[san] = {}
                continue
            analysis_board.push(move)
            info = engine.analyse(analysis_board, limit=limit)
            score: chess.engine.PovScore | None = info.get("score")
            if score is None:
                results[san] = {}
                continue
            mover_color = not analysis_board.turn
            pov_score = score.pov(mover_color)
            if pov_score.is_mate():
                results[san] = {"mate": pov_score.mate() or 0}
            else:
                cp = pov_score.score()
                results[san] = {"cp": int(cp)}
    return results

