"""
Fake Review Detection - Backend API
------------------------------------
This Flask app wraps the exact pipeline from DL_Fake_Review_Detection_Project_12.ipynb:
  1. Loads & cleans the dataset
  2. Runs the same EDA (label dist, rating dist, category dist, review length,
     boxplot, correlation heatmap)
  3. Trains the same LSTM model (Embedding -> LSTM(64) -> Dropout -> Dense(32) -> Dense(1))
  4. Evaluates it (accuracy/loss curves, confusion matrix, classification report)
  5. Serves everything to the frontend as JSON / base64 images
  6. Exposes a /api/predict endpoint for live single-review predictions

Everything is cached to disk after the first successful run (model, tokenizer,
graphs, metrics) so restarting the server is instant instead of retraining
every time.
"""

import os
import io
import json
import base64
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display backend needed on a server
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, request, jsonify
from flask_cors import CORS

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "fake reviews dataset.csv")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(CACHE_DIR, "model.keras")
TOKENIZER_PATH = os.path.join(CACHE_DIR, "tokenizer.pkl")
METRICS_PATH = os.path.join(CACHE_DIR, "metrics.json")
GRAPHS_PATH = os.path.join(CACHE_DIR, "graphs.json")

MAXLEN = 150
NUM_WORDS = 10000

app = Flask(__name__)
CORS(app)

# In-memory holders, populated by load_or_build() at startup
STATE = {
    "model": None,
    "tokenizer": None,
    "metrics": None,
    "graphs": None,
    "ready": False,
    "error": None,
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def fig_to_base64(fig):
    """Convert a matplotlib figure to a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")


def build_everything():
    """Runs the full notebook pipeline once and caches all artifacts."""

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            'Download "fake reviews dataset.csv" and place it in backend/data/'
        )

    # ---- Load & clean (mirrors notebook cells 3-14) ----
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates()
    df["text_"] = df["text_"].astype(str).str.lower()

    encoder = LabelEncoder()
    df["label"] = encoder.fit_transform(df["label"])
    # encoder.classes_ tells us what 0/1 actually mean, e.g. ['CG','OR'] -> fake/genuine
    class_names = [str(c) for c in encoder.classes_]

    df["review_length"] = df["text_"].apply(len)

    graphs = {}

    # ---- Graph 1: Label distribution ----
    fig = plt.figure(figsize=(6, 4))
    sns.countplot(x=df["label"])
    plt.title("Label Distribution")
    plt.xticks([0, 1], class_names)
    graphs["label_distribution"] = fig_to_base64(fig)

    # ---- Graph 2: Rating distribution ----
    fig = plt.figure(figsize=(6, 4))
    sns.countplot(x=df["rating"])
    plt.title("Rating Distribution")
    graphs["rating_distribution"] = fig_to_base64(fig)

    # ---- Graph 3: Category distribution ----
    fig = plt.figure(figsize=(12, 5))
    sns.countplot(y=df["category"], order=df["category"].value_counts().index)
    plt.title("Category Distribution")
    graphs["category_distribution"] = fig_to_base64(fig)

    # ---- Graph 4: Review length histogram ----
    fig = plt.figure(figsize=(8, 4))
    sns.histplot(df["review_length"], bins=40, kde=True)
    plt.title("Review Length Distribution")
    graphs["review_length_hist"] = fig_to_base64(fig)

    # ---- Graph 5: Boxplot review length by label ----
    fig = plt.figure(figsize=(6, 4))
    sns.boxplot(x=df["label"], y=df["review_length"])
    plt.xticks([0, 1], class_names)
    plt.title("Review Length by Label")
    graphs["boxplot_length_label"] = fig_to_base64(fig)

    # ---- Graph 6: Correlation heatmap ----
    fig = plt.figure(figsize=(5, 4))
    sns.heatmap(df[["rating", "label", "review_length"]].corr(), annot=True, cmap="Blues")
    plt.title("Correlation Heatmap")
    graphs["correlation_heatmap"] = fig_to_base64(fig)

    # ---- Tokenize (mirrors cell 24) ----
    X = df["text_"]
    y = df["label"]

    tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    X = pad_sequences(sequences, maxlen=MAXLEN, padding="post")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---- Model (mirrors cell 26) ----
    model = Sequential()
    model.add(Embedding(NUM_WORDS, 128, input_length=MAXLEN))
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(
        X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=2
    )

    # ---- Evaluation (mirrors cells 29-34) ----
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    prediction = model.predict(X_test, verbose=0)
    prediction = (prediction > 0.5).astype(int)

    report = classification_report(
        y_test, prediction, target_names=class_names, output_dict=True
    )
    cm = confusion_matrix(y_test, prediction)

    # ---- Graph 7: Confusion matrix ----
    fig = plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    graphs["confusion_matrix"] = fig_to_base64(fig)

    # ---- Graph 8: Accuracy curve ----
    fig = plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    graphs["accuracy_curve"] = fig_to_base64(fig)

    # ---- Graph 9: Loss curve ----
    fig = plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    graphs["loss_curve"] = fig_to_base64(fig)

    metrics = {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "classification_report": report,
        "class_names": class_names,
        "dataset_rows": int(df.shape[0]),
        "epochs": len(history.history["accuracy"]),
    }

    # ---- Persist everything ----
    model.save(MODEL_PATH)
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump({"tokenizer": tokenizer, "class_names": class_names}, f)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f)
    with open(GRAPHS_PATH, "w") as f:
        json.dump(graphs, f)

    return model, tokenizer, class_names, metrics, graphs


def load_or_build():
    """Loads cached artifacts if present, otherwise runs the full pipeline."""
    try:
        if all(os.path.exists(p) for p in [MODEL_PATH, TOKENIZER_PATH, METRICS_PATH, GRAPHS_PATH]):
            print("[startup] Cached model/graphs found - loading instantly...")
            model = load_model(MODEL_PATH)
            with open(TOKENIZER_PATH, "rb") as f:
                tok_data = pickle.load(f)
            tokenizer = tok_data["tokenizer"]
            class_names = tok_data["class_names"]
            with open(METRICS_PATH) as f:
                metrics = json.load(f)
            with open(GRAPHS_PATH) as f:
                graphs = json.load(f)
        else:
            print("[startup] No cache found - running full pipeline (this trains the LSTM, ~5-15 min on CPU)...")
            model, tokenizer, class_names, metrics, graphs = build_everything()

        STATE["model"] = model
        STATE["tokenizer"] = tokenizer
        STATE["class_names"] = class_names
        STATE["metrics"] = metrics
        STATE["graphs"] = graphs
        STATE["ready"] = True
        print("[startup] Ready! Model + graphs loaded.")
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
def graphs():
    if not STATE["ready"]:
        return jsonify({"error": STATE["error"] or "Model not ready yet"}), 503
    return jsonify(STATE["graphs"])


@app.route("/api/metrics")
def metrics():
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


import os

# Load model when app starts (works for Gunicorn too)
load_or_build()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)