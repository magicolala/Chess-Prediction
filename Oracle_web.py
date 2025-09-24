"""Flask application exposing the analysis pipeline via JSON API."""

from __future__ import annotations

import os
from typing import Callable, Dict

import chess.engine
from flask import Flask, jsonify, request
from flask_cors import CORS

from oracle.pipeline.analyze import analyze as default_analyze

DEFAULT_ELO = 1500
DEFAULT_TIME_CONTROL = "classical"
DEFAULT_STOCKFISH_PATH = os.getenv("STOCKFISH_PATH", "")

INDEX_HTML_TEMPLATE = """<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <title>Oracle Analyzer</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2rem; }
    textarea { width: 100%; height: 200px; }
    table { border-collapse: collapse; margin-top: 1rem; width: 100%; }
    th, td { border: 1px solid #ccc; padding: 0.5rem; text-align: left; }
    .controls { margin-top: 0.75rem; display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: center; }
    .controls label { font-weight: bold; }
    .controls input, .controls select { padding: 0.3rem; }
    .stockfish { flex: 1 1 100%; display: flex; flex-direction: column; }
    .stockfish input { width: 100%; box-sizing: border-box; }
  </style>
</head>
<body>
  <h1>Oracle Analyzer</h1>
  <label for='pgn'>PGN</label><br>
  <textarea id='pgn'></textarea>
  <div class='controls'>
    <label for='elo'>Elo:</label>
    <input id='elo' type='number' value='1500'>
    <label for='time_control'>Time Control:</label>
    <select id='time_control'>
      <option value='bullet'>bullet</option>
      <option value='blitz'>blitz</option>
      <option value='rapid' selected>rapid</option>
      <option value='classical'>classical</option>
    </select>
    <div class='stockfish'>
      <label for='stockfish_path'>Stockfish Path:</label>
      <input id='stockfish_path' type='text' placeholder='/usr/local/bin/stockfish' value='__STOCKFISH_PATH__'>
    </div>
    <button onclick='analyze()'>Analyze</button>
  </div>
  <div id='result'></div>
<script>
async function analyze() {
  const stockfishPath = document.getElementById('stockfish_path').value.trim();
  const payload = {
    pgn: document.getElementById('pgn').value,
    context: {
      elo: parseInt(document.getElementById('elo').value, 10),
      time_control: document.getElementById('time_control').value
    },
    stockfish_path: stockfishPath
  };
  const response = await fetch('/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  const container = document.getElementById('result');
  if (!response.ok) {
    let errorMessage = 'Request failed';
    try {
      const errorPayload = await response.json();
      if (errorPayload && errorPayload.error) {
        errorMessage = errorPayload.error;
      }
    } catch (err) {
      // ignore parsing errors and fall back to default message
    }
    container.innerHTML = `<p>${errorMessage}</p>`;
    return;
  }
  const data = await response.json();
  let html = `<p>Expected score: ${data.expected_score.toFixed(3)}</p>`;
  html += '<table><thead><tr><th>Move</th><th>Prior %</th><th>Adjusted %</th><th>SF Eval</th><th>Quality</th></tr></thead><tbody>';
  for (const move of data.moves) {
    html += `<tr><td>${move.san}</td><td>${move.prior_pct.toFixed(2)}</td><td>${move.adjusted_pct.toFixed(2)}</td><td>${move.sf_eval_cp ?? ''}</td><td>${move.quality ?? ''}</td></tr>`;
  }
  html += '</tbody></table>';
  container.innerHTML = html;
}
</script>
</body>
</html>
"""


def _make_engine_factory(path: str) -> Callable[[], chess.engine.SimpleEngine]:
    def factory() -> chess.engine.SimpleEngine:
        return chess.engine.SimpleEngine.popen_uci(path)

    return factory


def _normalize_time_control(value: str) -> str:
    value = (value or DEFAULT_TIME_CONTROL).lower()
    if value in {"bullet", "blitz", "rapid", "classical"}:
        return value
    return DEFAULT_TIME_CONTROL


def _parse_context(payload: Dict[str, object]) -> Dict[str, object]:
    context = payload.get("context") or {}
    elo_value = context.get("elo")
    try:
        elo = int(elo_value) if elo_value is not None else DEFAULT_ELO
    except (TypeError, ValueError):
        elo = DEFAULT_ELO
    time_control = _normalize_time_control(
        context.get("time_control", DEFAULT_TIME_CONTROL)
    )
    return {"elo": elo, "time_control": time_control}


def create_app(analyze_fn: Callable[..., Dict[str, object]] = default_analyze) -> Flask:
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}})

    @app.post("/api/analyze")
    def api_analyze():
        payload = request.get_json(force=True) or {}
        pgn = payload.get("pgn", "").strip()
        if not pgn:
            return jsonify({"error": "Missing PGN"}), 400
        ctx = _parse_context(payload)
        stockfish_path = (payload.get("stockfish_path") or "").strip()
        engine_factory = (
            _make_engine_factory(stockfish_path) if stockfish_path else None
        )
        try:
            result = analyze_fn(pgn=pgn, ctx=ctx, engine_factory=engine_factory)
        except (FileNotFoundError, chess.engine.EngineError) as exc:
            app.logger.exception("Stockfish engine error", exc_info=exc)
            return (
                jsonify({"error": "Le chemin Stockfish est invalide ou inaccessible."}),
                400,
            )
        except Exception as exc:  # noqa: BLE001
            app.logger.exception("Analysis failed", exc_info=exc)
            return jsonify({"error": "Erreur interne lors de l'analyse."}), 500
        return jsonify(result)

    @app.get("/")
    def index():
        return INDEX_HTML_TEMPLATE.replace("__STOCKFISH_PATH__", DEFAULT_STOCKFISH_PATH)

    return app


if __name__ == "__main__":
    host = os.getenv("ORACLE_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("ORACLE_WEB_PORT", "8000"))
    debug = os.getenv("ORACLE_WEB_DEBUG", "false").lower() == "true"
    create_app().run(host=host, port=port, debug=debug)
