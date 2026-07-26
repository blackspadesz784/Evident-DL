"""
Fake Review Detection - Backend API (PRODUCTION / INFERENCE ONLY)
--------------------------------------------------------------------
This is the lightweight version meant for Render deployment.

It does NOT train anything and does NOT import pandas / matplotlib /
seaborn / scikit-learn. It only:
  1. Loads the already-trained model + tokenizer + cached graphs/metrics
     from the local `cache/` folder (generated earlier by train.py)
  2. Serves them to the frontend as JSON
  3. Exposes /api/predict for live single-review predictions

BEFORE DEPLOYING TO RENDER:
  1. Run `python train.py` on your own machine (full dataset needed
     there). This creates a `cache/` folder containing:
        cache/model.keras
        cache/tokenizer.pkl
        cache/metrics.json
        cache/graphs.json
  2. Copy that entire `cache/` folder next to this app.py and push/
     upload it to your Render repo. Do NOT upload the dataset CSV to
     Render — it isn't needed anymore and just wastes disk space.
  3. Deploy. Startup will be instant since nothing is trained here.
"""

import os
import json
import pickle

from flask import Flask, request, jsonify
from flask_cors import CORS

# Only needed so pickle can reconstruct the Tokenizer object, and for
# padding sequences / loading the saved model. No pandas/sklearn/plots.
from tensorflow.keras.preprocessing.text import Tokenizer  # noqa: F401 (needed for unpickling)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")

MODEL_PATH = os.path.join(CACHE_DIR, "model.keras")
TOKENIZER_PATH = os.path.join(CACHE_DIR, "tokenizer.pkl")
METRICS_PATH = os.path.join(CACHE_DIR, "metrics.json")
GRAPHS_PATH = os.path.join(CACHE_DIR, "graphs.json")

MAXLEN = 150

app = Flask(__name__)
CORS(app)

STATE = {
    "model": None,
    "tokenizer": None,
    "class_names": None,
    "metrics": None,
    "graphs": None,
    "ready": False,
    "error": None,
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def load_cached_artifacts():
    """Loads the pre-trained model + tokenizer + graphs + metrics.
    Raises if any file is missing — production never trains, it only serves."""
    missing = [
        p for p in [MODEL_PATH, TOKENIZER_PATH, METRICS_PATH, GRAPHS_PATH]
        if not os.path.exists(p)
    ]
    if missing:
        raise FileNotFoundError(
            "Missing cached artifact(s): " + ", ".join(missing) +
            ". Train locally with train.py first, then upload the cache/ "
            "folder alongside app.py."
        )

    model = load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "rb") as f:
        tok_data = pickle.load(f)
    tokenizer = tok_data["tokenizer"]
    class_names = tok_data["class_names"]

    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    with open(GRAPHS_PATH) as f:
        graphs = json.load(f)

    return model, tokenizer, class_names, metrics, graphs


def load_or_fail():
    """Loads cached artifacts at startup. Never trains on Render."""
    try:
        print("[startup] Loading cached model/graphs (production mode, no training)...")
        model, tokenizer, class_names, metrics, graphs = load_cached_artifacts()

        STATE["model"] = model
        STATE["tokenizer"] = tokenizer
        STATE["class_names"] = class_names
        STATE["metrics"] = metrics
        STATE["graphs"] = graphs
        STATE["ready"] = True
        print("[startup] Ready! Model + graphs loaded from cache.")
    except Exception as e:
        STATE["error"] = str(e)
        print(f"[startup] ERROR: {e}")


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/api/status")
def status():
    return jsonify({"ready": STATE["ready"], "error": STATE["error"]})


@app.route("/api/graphs")
def graphs_route():
    if not STATE["ready"]:
        return jsonify({"error": STATE["error"] or "Model not ready yet"}), 503
    return jsonify(STATE["graphs"])


@app.route("/api/metrics")
def metrics_route():
    if not STATE["ready"]:
        return jsonify({"error": STATE["error"] or "Model not ready yet"}), 503
    return jsonify(STATE["metrics"])


@app.route("/api/predict", methods=["POST"])
def predict():
    if not STATE["ready"]:
        return jsonify({"error": STATE["error"] or "Model not ready yet"}), 503

    data = request.get_json(force=True)
    review = (data or {}).get("review", "").strip()
    if not review:
        return jsonify({"error": "Please provide a non-empty 'review' string"}), 400

    tokenizer = STATE["tokenizer"]
    model = STATE["model"]
    class_names = STATE["class_names"]

    seq = tokenizer.texts_to_sequences([review.lower()])
    pad = pad_sequences(seq, maxlen=MAXLEN, padding="post")
    pred = model.predict(pad, verbose=0)
    score = float(pred[0][0])

    label_idx = 1 if score > 0.5 else 0
    label_name = class_names[label_idx] if label_idx < len(class_names) else str(label_idx)
    is_genuine = score > 0.5

    return jsonify({
        "review": review,
        "score": score,
        "label_index": label_idx,
        "label_name": label_name,
        "prediction": "Genuine Review" if is_genuine else "Fake Review",
        "confidence": score if is_genuine else 1 - score,
    })


# Load model when app starts (works for Gunicorn too)
load_or_fail()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)