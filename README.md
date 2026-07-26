# Evident — Frontend + Backend

This turns your notebook (`DL_Fake_Review_Detection_Project_12.ipynb`) into a
running web app:

- **backend/app.py** — Flask API that runs the *exact same* pipeline as your
  notebook (cleaning → EDA → LSTM training → evaluation → prediction) and
  serves the results as JSON/images.
- **frontend/index.html** — A single-page dashboard that calls the backend
  and displays every graph from the notebook, plus a live "type a review, get
  a verdict" tool.

```
fake-review-app/
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   └── data/               ← put the dataset CSV here
└── frontend/
    └── index.html
```

## Step 1 — Install Python & get the project

Make sure you have **Python 3.10 or 3.11** installed (TensorFlow doesn't yet
support 3.12+ well). Check with:

```bash
python3 --version
```

Unzip the project you downloaded, then open a terminal inside it.

## Step 2 — Get the dataset

Your notebook reads `/content/fake reviews dataset.csv` (a Colab path). Download
the same CSV (search "Fake Reviews Dataset" — the common one is on Kaggle:
`fake-reviews-dataset` by mexwell / lievgarcia) and place it at:

```
fake-review-app/backend/data/fake reviews dataset.csv
```

(Keep the filename exactly as shown, spaces included — it matches what
`app.py` looks for.)

## Step 3 — Set up the backend

```bash
cd fake-review-app/backend
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

This installs Flask, TensorFlow, pandas, scikit-learn, matplotlib, seaborn, etc.
(TensorFlow is a big download — it can take a few minutes.)

## Step 4 — Run the backend

```bash
python app.py
```

- **First run:** it trains the LSTM model from scratch (same as your notebook,
  10 epochs) — this can take anywhere from a few minutes to ~20-30 minutes on
  a normal CPU, depending on dataset size. You'll see epoch-by-epoch progress
  in the terminal, same as Colab.
- Once done, it **caches** the trained model, tokenizer, and all 9 graphs
  inside `backend/cache/`. Every future run will just load the cache and start
  in a couple seconds — no retraining.
- Keep this terminal window open — it's your API server running at
  `https://evident-dl.onrender.com`.

## Step 5 — Open the frontend

Just open `frontend/index.html` directly in your browser (double-click it, or
right-click → Open With → your browser).

You'll see:
- A status pill at top right — turns **green** once the backend is ready.
- A live "Analyze a review" box — type/paste any review and click **Analyze
  Review** to get a Genuine/Fake verdict with a confidence score.
- All 9 graphs from your notebook (label distribution, rating distribution,
  category distribution, review length histogram, boxplot, correlation
  heatmap, confusion matrix, accuracy curve, loss curve).
- A classification report table (precision/recall/F1 per class).

> If you see "Cannot reach backend" in the status pill, make sure
> `python app.py` is still running in your terminal, and that nothing else is
> using port 5000.

## Step 6 — (Optional) Retrain from scratch

If you change the dataset or model code and want to retrain, just delete the
cache folder and restart the backend:

```bash
rm -rf backend/cache
python app.py
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: Dataset not found` | Double-check the CSV is at `backend/data/fake reviews dataset.csv` with that exact name. |
| Port 5000 already in use | Edit the last line of `app.py`: change `port=5000` to e.g. `port=5050`, and update `API_BASE` in `frontend/index.html` to match. |
| TensorFlow install fails | Make sure you're on Python 3.10/3.11, not 3.12+. |
| CORS error in browser console | Confirm `flask-cors` installed correctly (`pip show flask-cors`) — it's already wired into `app.py`. |
| Training very slow | Expected on CPU — LSTMs are slow without a GPU. It only happens once thanks to caching. |
