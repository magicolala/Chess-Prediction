"""Flask application exposing the analysis pipeline via JSON API."""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional, Tuple

import chess.engine
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

from oracle.llm.base import SequenceProvider
from oracle.llm.llama_cpp_local import LlamaCppLocalProvider
from oracle.llm.selector import DEFAULT_TOP_K
from oracle.llm.transformers_local import TransformersLocalProvider
from oracle.pipeline.analyze import analyze as default_analyze

load_dotenv()
DEFAULT_ELO = 1500
DEFAULT_TIME_CONTROL = "classical"
DEFAULT_STOCKFISH_PATH = os.getenv("STOCKFISH_PATH", "")
DEFAULT_LLM_BACKEND = os.getenv("ORACLE_LLM_BACKEND", "")
DEFAULT_TRANSFORMERS_MODEL_ID = os.getenv("ORACLE_MODEL_ID", "")
DEFAULT_LLAMA_MODEL_PATH = os.getenv("ORACLE_GGUF_PATH", "")


def _coerce_numeric_env(name: str, fallback: str, cast) -> str:
    value = os.getenv(name)
    if value is None:
        return fallback
    try:
        cast(value)
    except (TypeError, ValueError):
        return fallback
    return str(value)


DEFAULT_LLM_DEPTH = _coerce_numeric_env("ORACLE_LLM_DEPTH", "3", int)
DEFAULT_LLM_TOP_K = _coerce_numeric_env("ORACLE_TOP_K", str(DEFAULT_TOP_K), int)
DEFAULT_LLM_PROB_THRESHOLD = _coerce_numeric_env(
    "ORACLE_PROB_THRESHOLD", "0.001", float
)

_PROVIDER_CACHE: Dict[Tuple[str, ...], SequenceProvider] = {}

INDEX_HTML_TEMPLATE = """<!doctype html>
<html lang='fr'>
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
    .llm-config { margin-top: 1.5rem; border: 1px solid #e0e0e0; padding: 1rem; border-radius: 6px; background: #fafafa; }
    .llm-config h2 { margin-top: 0; font-size: 1.1rem; }
    .llm-config label { font-weight: bold; margin-top: 0.5rem; }
    .llm-config input, .llm-config select { margin-top: 0.25rem; }
    .llm-fields { display: none; flex-direction: column; gap: 0.5rem; margin-top: 0.5rem; }
    .llm-fields input { width: 100%; box-sizing: border-box; }
    .llm-advanced { margin-top: 0.75rem; display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 0.5rem 1rem; align-items: end; }
    .llm-advanced label { font-weight: bold; }
    button.primary { padding: 0.5rem 1rem; font-size: 1rem; cursor: pointer; }
  </style>
</head>
<body>
  <h1>Oracle Analyzer</h1>
  <label for='pgn'>PGN</label><br>
  <textarea id='pgn'></textarea>
  <div class='controls'>
    <label for='elo'>Elo :</label>
    <input id='elo' type='number' value='1500'>
    <label for='time_control'>Cadence :</label>
    <select id='time_control'>
      <option value='bullet'>bullet</option>
      <option value='blitz'>blitz</option>
      <option value='rapid' selected>rapid</option>
      <option value='classical'>classical</option>
    </select>
    <div class='stockfish'>
      <label for='stockfish_path'>Chemin Stockfish :</label>
      <input id='stockfish_path' type='text' placeholder='/usr/local/bin/stockfish' value='__STOCKFISH_PATH__'>
    </div>
    <button class='primary' onclick='analyze()'>Analyser</button>
  </div>
  <div class='llm-config'>
    <h2>Configuration LLM</h2>
    <label for='llm_backend'>Backend :</label>
    <select id='llm_backend' onchange='updateLlmFields()'>
      <option value=''>Défaut (variables d'environnement)</option>
      <option value='transformers'>Transformers (local)</option>
      <option value='llama_cpp'>llama.cpp (GGUF)</option>
    </select>
    <div class='llm-fields' id='llm_transformers'>
      <label for='llm_model_id'>Modèle Transformers :</label>
      <input id='llm_model_id' type='text' value='__TRANSFORMERS_MODEL_ID__' placeholder='ex: meta-llama/Llama-3-8B-Instruct'>
      <label for='llm_hf_token'>Jeton Hugging Face (optionnel) :</label>
      <input id='llm_hf_token' type='password' value=''>
    </div>
    <div class='llm-fields' id='llm_llama'>
      <label for='llm_model_path'>Chemin du modèle GGUF :</label>
      <input id='llm_model_path' type='text' value='__LLAMA_MODEL_PATH__' placeholder='/chemin/vers/modele.gguf'>
    </div>
    <div class='llm-advanced'>
      <div>
        <label for='llm_depth'>Profondeur :</label>
        <input id='llm_depth' type='number' min='1' max='8' value='__LLM_DEPTH__'>
      </div>
      <div>
        <label for='llm_top_k'>Top K :</label>
        <input id='llm_top_k' type='number' min='1' max='20' value='__LLM_TOP_K__'>
      </div>
      <div>
        <label for='llm_prob_threshold'>Seuil de probabilité :</label>
        <input id='llm_prob_threshold' type='number' min='0.000001' max='0.5' step='0.0001' value='__LLM_PROB_THRESHOLD__'>
      </div>
    </div>
  </div>
  <div id='result'></div>
<script>
const DEFAULT_BACKEND = '__LLM_BACKEND__';
function updateLlmFields() {
  const backend = document.getElementById('llm_backend').value;
  const tf = document.getElementById('llm_transformers');
  const llama = document.getElementById('llm_llama');
  if (backend === 'transformers') {
    tf.style.display = 'flex';
    llama.style.display = 'none';
  } else if (backend === 'llama_cpp') {
    tf.style.display = 'none';
    llama.style.display = 'flex';
  } else {
    tf.style.display = 'none';
    llama.style.display = 'none';
  }
}
function collectLlmConfig() {
  const backend = document.getElementById('llm_backend').value;
  const config = {};
  const depthValue = parseInt(document.getElementById('llm_depth').value, 10);
  if (!Number.isNaN(depthValue)) {
    config.depth = depthValue;
  }
  const topKValue = parseInt(document.getElementById('llm_top_k').value, 10);
  if (!Number.isNaN(topKValue)) {
    config.top_k = topKValue;
  }
  const probThresholdValue = parseFloat(document.getElementById('llm_prob_threshold').value);
  if (!Number.isNaN(probThresholdValue)) {
    config.prob_threshold = probThresholdValue;
  }
  if (backend === 'transformers') {
    config.backend = backend;
    const modelId = document.getElementById('llm_model_id').value.trim();
    if (modelId) {
      config.model_id = modelId;
    }
    const hfToken = document.getElementById('llm_hf_token').value.trim();
    if (hfToken) {
      config.token = hfToken;
    }
  } else if (backend === 'llama_cpp') {
    config.backend = backend;
    const modelPath = document.getElementById('llm_model_path').value.trim();
    if (modelPath) {
      config.model_path = modelPath;
    }
  } else if (Object.keys(config).length === 0) {
    return null;
  }
  return Object.keys(config).length > 0 ? config : null;
}
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
  const llmConfig = collectLlmConfig();
  if (llmConfig) {
    payload.llm = llmConfig;
  }
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
document.addEventListener('DOMContentLoaded', () => {
  if (DEFAULT_BACKEND) {
    document.getElementById('llm_backend').value = DEFAULT_BACKEND;
  }
  updateLlmFields();
  console.log('--- Current Configuration ---');
  console.log('STOCKFISH_PATH: ', document.getElementById('stockfish_path').value);
  console.log('LLM_BACKEND: ', document.getElementById('llm_backend').value || DEFAULT_BACKEND);
  console.log('TRANSFORMERS_MODEL_ID: ', document.getElementById('llm_model_id').value);
  console.log('LLAMA_MODEL_PATH: ', document.getElementById('llm_model_path').value);
  console.log('LLM_DEPTH: ', document.getElementById('llm_depth').value);
  console.log('LLM_TOP_K: ', document.getElementById('llm_top_k').value);
  console.log('LLM_PROB_THRESHOLD: ', document.getElementById('llm_prob_threshold').value);
  console.log('-----------------------------');
});
</script>
</body>
</html>
"""


def _clean_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_backend(value: Any) -> str:
    backend = _clean_string(value).lower()
    if not backend:
        raise ValueError("Merci de sélectionner un backend LLM.")
    if backend not in {"transformers", "llama_cpp"}:
        raise ValueError(
            "Backend LLM inconnu. Choisissez 'transformers' ou 'llama_cpp'."
        )
    return backend


def _provider_cache_key(options: Dict[str, Any]) -> Tuple[str, ...]:
    backend = _normalize_backend(options.get("backend"))
    if backend == "transformers":
        model_id = _clean_string(options.get("model_id"))
        token = _clean_string(options.get("token"))
        return (backend, model_id, token)
    if backend == "llama_cpp":
        model_path = _clean_string(options.get("model_path"))
        return (backend, model_path)
    return (backend,)


def _build_provider(options: Dict[str, Any]) -> SequenceProvider:
    backend = _normalize_backend(options.get("backend"))
    if backend == "transformers":
        model_id = _clean_string(options.get("model_id"))
        if not model_id:
            raise ValueError("Merci d'indiquer l'identifiant du modèle Transformers.")
        token = _clean_string(options.get("token"))
        if token:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
            os.environ["HF_HUB_TOKEN"] = token
        return TransformersLocalProvider(model_id=model_id)
    if backend == "llama_cpp":
        model_path = _clean_string(options.get("model_path"))
        if not model_path:
            raise ValueError("Merci d'indiquer le chemin du modèle GGUF.")
        return LlamaCppLocalProvider(model_path=model_path)
    raise ValueError("Backend LLM inconnu. Choisissez 'transformers' ou 'llama_cpp'.")


def _get_provider(options: Dict[str, Any]) -> SequenceProvider:
    key = _provider_cache_key(options)
    provider = _PROVIDER_CACHE.get(key)
    if provider is not None:
        return provider
    provider = _build_provider(options)
    _PROVIDER_CACHE[key] = provider
    return provider


def _parse_int(
    value: Any,
    label: str,
    *,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} doit être un entier.") from exc
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{label} doit être supérieur ou égal à {minimum}.")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{label} doit être inférieur ou égal à {maximum}.")
    return parsed


def _parse_float(
    value: Any,
    label: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} doit être un nombre.") from exc
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{label} doit être supérieur ou égal à {minimum}.")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{label} doit être inférieur ou égal à {maximum}.")
    return parsed


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
        llm_payload = payload.get("llm") or {}
        provider: Optional[SequenceProvider] = None
        analyze_overrides: Dict[str, Any] = {}
        backend_value = _clean_string(llm_payload.get("backend"))
        if backend_value:
            llm_payload["backend"] = backend_value
            try:
                provider = _get_provider(llm_payload)
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400
            except ImportError as exc:
                app.logger.exception("Missing LLM dependency", exc_info=exc)
                return (
                    jsonify(
                        {
                            "error": (
                                "Le backend LLM requis n'est pas disponible (dépendance manquante)."
                            )
                        }
                    ),
                    400,
                )
        try:
            depth = _parse_int(
                llm_payload.get("depth"),
                "La profondeur LLM",
                minimum=1,
                maximum=16,
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        if depth is not None:
            analyze_overrides["depth"] = depth

        try:
            top_k = _parse_int(
                llm_payload.get("top_k"),
                "Le paramètre Top K",
                minimum=1,
                maximum=50,
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        if top_k is not None:
            analyze_overrides["top_k"] = top_k

        try:
            prob_threshold = _parse_float(
                llm_payload.get("prob_threshold"),
                "Le seuil de probabilité",
                minimum=0.000001,
                maximum=0.5,
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        if prob_threshold is not None:
            analyze_overrides["prob_threshold"] = prob_threshold

        try:
            result = analyze_fn(
                pgn=pgn,
                ctx=ctx,
                provider=provider,
                engine_factory=engine_factory,
                **analyze_overrides,
            )
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
        print("--- Current Configuration ---")
        print(f"STOCKFISH_PATH: {DEFAULT_STOCKFISH_PATH}")
        print(f"LLM_BACKEND: {DEFAULT_LLM_BACKEND}")
        print(f"TRANSFORMERS_MODEL_ID: {DEFAULT_TRANSFORMERS_MODEL_ID}")
        print(f"LLAMA_MODEL_PATH: {DEFAULT_LLAMA_MODEL_PATH}")
        print(f"LLM_DEPTH: {DEFAULT_LLM_DEPTH}")
        print(f"LLM_TOP_K: {DEFAULT_LLM_TOP_K}")
        print(f"LLM_PROB_THRESHOLD: {DEFAULT_LLM_PROB_THRESHOLD}")
        print("-----------------------------")
        return (
            INDEX_HTML_TEMPLATE.replace("__STOCKFISH_PATH__", DEFAULT_STOCKFISH_PATH)
            .replace("__LLM_BACKEND__", DEFAULT_LLM_BACKEND)
            .replace("__TRANSFORMERS_MODEL_ID__", DEFAULT_TRANSFORMERS_MODEL_ID)
            .replace("__LLAMA_MODEL_PATH__", DEFAULT_LLAMA_MODEL_PATH)
            .replace("__LLM_DEPTH__", DEFAULT_LLM_DEPTH)
            .replace("__LLM_TOP_K__", DEFAULT_LLM_TOP_K)
            .replace("__LLM_PROB_THRESHOLD__", DEFAULT_LLM_PROB_THRESHOLD)
        )

    return app


if __name__ == "__main__":
    host = os.getenv("ORACLE_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("ORACLE_WEB_PORT", "8000"))
    debug = os.getenv("ORACLE_WEB_DEBUG", "false").lower() == "true"
    create_app().run(host=host, port=port, debug=debug)
