"""
Flask backend for the Zalo-like mobile vishing demo.

Endpoints:
  GET  /                    — serve the mobile demo UI
  GET  /api/scenarios       — list all demo call scenarios
  GET  /api/scenario/<id>   — full scenario data (chunks + metadata)
  POST /api/predict         — run vishing inference on a text chunk
  GET  /api/health          — health check / detector mode info
"""
from __future__ import annotations
import os
import sys
import logging
from pathlib import Path

from flask import Flask, render_template, jsonify, request

# Make project root importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from demo_scripts import get_scenarios_summary, get_scenario, get_all_scenario_ids
from inference import VishingDetector

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
detector = VishingDetector(project_root=str(PROJECT_ROOT))


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    scenarios = get_scenarios_summary()
    return render_template("index.html", scenarios=scenarios)


@app.route("/api/scenarios")
def api_scenarios():
    return jsonify(get_scenarios_summary())


@app.route("/api/scenario/<scenario_id>")
def api_scenario(scenario_id: str):
    s = get_scenario(scenario_id)
    if not s:
        return jsonify({"error": "not found"}), 404
    return jsonify(s)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True, silent=True) or {}
    text            = data.get("text", "").strip()
    cumulative_text = data.get("cumulative_text", "").strip()

    if not text:
        return jsonify({"error": "text is required"}), 400

    result = detector.predict(text, cumulative_text)
    return jsonify(result)


@app.route("/api/health")
def api_health():
    return jsonify({
        "status":   "ok",
        "detector": detector.mode,
        "scenarios": get_all_scenario_ids(),
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting Zalo Demo on http://localhost:%d", port)
    app.run(host="0.0.0.0", port=port, debug=True)
