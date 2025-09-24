"""Flask application exposing the analysis pipeline via JSON API."""

from __future__ import annotations

import os
from typing import Callable, Dict, Optional

from flask import Flask, jsonify, request
from flask_cors import CORS

from oracle.pipeline.analyze import analyze as default_analyze

DEFAULT_ELO = 1500
DEFAULT_TIME_CONTROL = "classical"


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
    time_control = _normalize_time_control(context.get("time_control", DEFAULT_TIME_CONTROL))
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
        result = analyze_fn(pgn=pgn, ctx=ctx)
        return jsonify(result)

    @app.get("/")
    def index():
        return (
            """<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <title>Oracle Analyzer</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2rem; }
    textarea { width: 100%; height: 200px; }
    table { border-collapse: collapse; margin-top: 1rem; width: 100%; }
    th, td { border: 1px solid #ccc; padding: 0.5rem; text-align: left; }
  </style>
</head>
<body>
  <h1>Oracle Analyzer</h1>
  <label for='pgn'>PGN</label><br>
  <textarea id='pgn'></textarea>
  <div>
    <label for='elo'>Elo:</label>
    <input id='elo' type='number' value='1500'>
    <label for='time_control'>Time Control:</label>
    <select id='time_control'>
      <option value='bullet'>bullet</option>
      <option value='blitz'>blitz</option>
      <option value='rapid' selected>rapid</option>
      <option value='classical'>classical</option>
    </select>
    <button onclick='analyze()'>Analyze</button>
  </div>
  <div id='result'></div>
<script>
async function analyze() {
  const payload = {
    pgn: document.getElementById('pgn').value,
    context: {
      elo: parseInt(document.getElementById('elo').value, 10),
      time_control: document.getElementById('time_control').value
    }
  };
  const response = await fetch('/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  const container = document.getElementById('result');
  if (!response.ok) {
    container.innerHTML = '<p>Request failed</p>';
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
        )

    return app


if __name__ == "__main__":
    host = os.getenv("ORACLE_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("ORACLE_WEB_PORT", "8000"))
    debug = os.getenv("ORACLE_WEB_DEBUG", "false").lower() == "true"
    create_app().run(host=host, port=port, debug=debug)
