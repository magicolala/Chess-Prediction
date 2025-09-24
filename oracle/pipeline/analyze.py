"""High-level analysis pipeline combining LLM and Stockfish signals."""

from __future__ import annotations

import io
from typing import Any, Dict, List, Mapping, Optional

import chess
import chess.pgn

from oracle.calib.calibrator import Calibrator
from oracle.calib.quality import classify_quality, load_thresholds
from oracle.core.expected import aggregate_expected_score
from oracle.core.expand import expand_san_candidates
from oracle.llm import selector
from oracle.sf.adapter import evaluate_moves


def _board_from_pgn(pgn: str) -> chess.Board:
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        raise ValueError("Invalid PGN supplied for analysis")
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
    return board


def _evaluation_cp(result: Mapping[str, Any]) -> Optional[float]:
    if result is None:
        return None
    if "mate" in result and result["mate"] is not None:
        mate = result["mate"]
        if mate is None:
            return None
        return 10000.0 if mate > 0 else -10000.0
    if "cp" in result and result["cp"] is not None:
        return float(result["cp"])
    return None


def analyze(
    pgn: str,
    ctx: Mapping[str, Any],
    *,
    provider=None,
    engine_factory=None,
    calibrator: Optional[Calibrator] = None,
    depth: int = 3,
    prob_threshold: float = 0.0,
    top_k: Optional[int] = None,
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full prediction and calibration pipeline."""

    board = _board_from_pgn(pgn)
    legal_moves = {board.san(move) for move in board.legal_moves}

    if provider is None:
        provider = selector.build_sequence_provider()
    if top_k is None:
        top_k = selector.get_top_k()

    base_prompt = prompt if prompt is not None else pgn

    prior_sequences = expand_san_candidates(
        provider=provider,
        prompt=base_prompt,
        legal_moves=legal_moves,
        depth=depth,
        prob_threshold=prob_threshold,
        top_k=top_k,
    )

    moves: List[Dict[str, Any]] = [
        {"san": san, "prior_logprob": logprob} for san, logprob in prior_sequences
    ]

    engine_results: Dict[str, Mapping[str, Any]] = {}
    if engine_factory is not None and moves:
        san_list = [move["san"] for move in moves]
        engine_results = evaluate_moves(board, san_list, engine_factory)

    thresholds = load_thresholds()

    best_cp: Optional[float] = None
    for result in engine_results.values():
        cp_equivalent = _evaluation_cp(result)
        if cp_equivalent is None:
            continue
        if best_cp is None or cp_equivalent > best_cp:
            best_cp = cp_equivalent

    for move in moves:
        san = move["san"]
        eval_result = engine_results.get(san, {})
        if "cp" in eval_result:
            move["sf_eval_cp"] = eval_result["cp"]
        if "mate" in eval_result:
            move["sf_eval_mate"] = eval_result["mate"]
        cp_value = _evaluation_cp(eval_result)
        if best_cp is None or cp_value is None:
            move["quality"] = "unknown"
        else:
            move["quality"] = classify_quality(best_cp, cp_value, thresholds)

    calibrator = calibrator or Calibrator()
    calibrated = calibrator.adjust(moves, context=ctx)

    move_lookup = {move["san"]: move for move in moves}
    for entry in calibrated:
        original = move_lookup.get(entry["san"], {})
        if "sf_eval_cp" in original:
            entry["sf_eval_cp"] = original["sf_eval_cp"]
        if "sf_eval_mate" in original:
            entry["sf_eval_mate"] = original["sf_eval_mate"]
        entry["quality"] = original.get("quality", entry.get("quality"))

    expected_score = aggregate_expected_score(calibrated)

    model_name = getattr(provider, "model_id", provider.__class__.__name__)

    return {
        "model": model_name,
        "moves": calibrated,
        "expected_score": expected_score,
        "usage": {"top_k": top_k, "depth": depth},
    }

