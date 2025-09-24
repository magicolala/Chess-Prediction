import csv
import importlib.machinery
import types
from pathlib import Path
from textwrap import dedent


def _load_module(path: str, name: str):
    loader = importlib.machinery.SourceFileLoader(name, path)
    module = types.ModuleType(name)
    loader.exec_module(module)
    return module


cli_pgn = _load_module("Oracle_pgn_file", "oracle_pgn_file")


def test_pgn_file_writes_csv(tmp_path: Path):
    pgn_content = dedent(
        """
        [Event "Game 1"]

        1. e4 e5

        [Event "Game 2"]

        1. d4 d5
        """
    ).strip()
    pgn_path = tmp_path / "games.pgn"
    pgn_path.write_text(pgn_content)
    output_path = tmp_path / "out.csv"

    calls = []

    def fake_analyze(**kwargs):
        calls.append(kwargs["pgn"])
        return {
            "model": "fake",
            "expected_score": 0.5,
            "usage": {"top_k": kwargs.get("top_k", 2)},
            "moves": [
                {
                    "san": "Nf3",
                    "prior_pct": 55.0,
                    "adjusted_pct": 60.0,
                    "sf_eval_cp": 80,
                    "quality": "good",
                }
            ],
        }

    cli_pgn.main(
        pgn_path=str(pgn_path),
        output_path=str(output_path),
        analyze_fn=fake_analyze,
        depth=2,
        prob_threshold=0.0,
        top_k=2,
    )

    assert len(calls) == 2

    with output_path.open() as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == [
        "game_index",
        "move_index",
        "san",
        "prior_pct",
        "adjusted_pct",
        "sf_eval_cp",
        "quality",
    ]
    assert rows[0]["san"] == "Nf3"
    assert rows[0]["prior_pct"] == "55.0"
    assert rows[0]["adjusted_pct"] == "60.0"
