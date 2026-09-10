# Usage details

Always run commands from the **repo root**.

## Pipeline orchestrator

`run_pipeline.py` runs: **scrape (optional) → clean → features → train (optional)**.

```bash
.venv/bin/python run_pipeline.py [options]
.venv/bin/python run_pipeline.py -h
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--skip-scrape` | off | Reuse `data/raw/` |
| `--skip-train` | off | Stop after features |
| `--start-page N` | `1` | First Immobiliare page |
| `--max-pages N` | `1` | Pages from `--start-page` |
| `--no-html` | off | Do not save raw HTML |
| `--no-mlflow` | off | Skip MLflow logging (still writes `models/`) |
| `-v` | off | Verbose logs |

Examples:

```bash
.venv/bin/python run_pipeline.py -v
.venv/bin/python run_pipeline.py --skip-scrape -v
.venv/bin/python run_pipeline.py --skip-scrape --skip-train -v
.venv/bin/python run_pipeline.py --max-pages 5 --no-html --no-mlflow -v
```

## Daily cadence (GitHub Actions)

Workflow [`.github/workflows/daily_monitoring.yml`](../.github/workflows/daily_monitoring.yml):

- **Schedule:** every day 06:00 UTC (`0 6 * * *`)
- **Manual:** Actions → *daily-monitoring* → *Run workflow*
- **Steps:** `run_pipeline.py --max-pages 100 --no-html --skip-train` → `etl.prune_raw --keep 1` → DVC push → `ml.drift_report` → `ml.retrain_check` (train **only** if MAE gate fires)
- **Storage:** no HTML; raw is a **rolling window** (newest scrape only). Listing history is merged into `features_latest.jsonl` (DVC).
- **Artifacts:** `reports/drift_latest/`, `reports/retrain_latest/`, `metrics.json`, `dataset.json` (14 days; no raw listings in git)

Unit tests on push/PR: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).

Note: scrape from GitHub-hosted runners may be blocked by the site; if the job fails on extract, re-run locally or use a self-hosted runner.

## Official stage commands

```bash
.venv/bin/python etl/extract/immobiliare_scraper.py --max-pages 5 -v
.venv/bin/python -m etl.transform.clean_phase -v
.venv/bin/python -m etl.transform.features_phase -v
.venv/bin/python -m ml.train -v
.venv/bin/python -m ml.drift_report -v
.venv/bin/python -m ml.retrain_check --dry-run -v
```
Train only:

```bash
.venv/bin/python -m ml.train --input data/processed/features_latest.jsonl -v
.venv/bin/python -m ml.train --no-mlflow -v
```

## Dataset versioning (RF-12)

Each train run fingerprints the input JSONL (SHA-256 + size + row count) and writes `dataset.json` next to the model. The same block is embedded in `metrics.json` under `dataset`, and MLflow logs `dataset_sha256` + the artifact. Listing files stay gitignored (RNF-03). Optional private sync across CI/machines: [`dvc.md`](dvc.md) (DVC + R2/S3).

```bash
cat models/baseline_latest/dataset.json
```

## Outputs

| Stage | Path |
|-------|------|
| Raw scrape | `data/raw/immobiliare_roma_<ts>/` |
| Clean + quality | `data/processed/listings_*.jsonl`, `rejected_*.jsonl`, `quality_*.json` |
| Features | `data/processed/features_*.jsonl` (+ `features_latest.jsonl`) |
| Model | `models/baseline_<ts>/`, `models/baseline_latest/` (`model.joblib`, `metrics.json`, `dataset.json`) |
| Drift | `reports/drift_<ts>/`, `reports/drift_latest/` (`report.html` + `summary.json`) |
| Retrain gate | `reports/retrain_<ts>/`, `reports/retrain_latest/` (`decision.json`) |
| Dashboard | `streamlit run dashboard/app.py` (local) |
| MLflow | `mlflow.db` (SQLite) |

## MLflow UI

Training uses SQLite (`mlflow.db`). Prefer this over `./mlruns` with MLflow 3.

```bash
source .venv/bin/activate
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) → experiment **`roma-rent-baseline`** → **Runs** (not Overview / Traces) → Metrics + Artifacts.

**Traces = 0** is expected (classic training runs only).

One-time migrate from an old FileStore folder:

```bash
.venv/bin/mlflow migrate-filestore --source ./mlruns --target sqlite:///mlflow.db
```

## Drift report (Evidently)

Compares **reference** vs **current** on `features_*.jsonl` using the same temporal/holdout split as training. If `models/baseline_latest/model.joblib` exists, adds a `prediction` column (prediction drift). HTML + JSON under `reports/`.

Full explanation (what Evidently is, step-by-step code flow, how drift is measured): [`drift.md`](drift.md).

```bash
.venv/bin/python -m ml.drift_report -v
.venv/bin/python -m ml.drift_report --input data/processed/features_latest.jsonl -v
.venv/bin/python -m ml.drift_report --no-model -v
```

Open `reports/drift_latest/report.html`. Summary metrics: `drifted_columns_count` / `drifted_columns_share` in `summary.json`; with a model, also `mae_reference` / `mae_current` and `mae_by_day`.

## Retrain gate (RF-09)

Reads `reports/drift_latest/summary.json` and retrains if `mae_current / mae_reference >= 1.5` and `n_reference >= 50`. Feature/prediction drift is logged in the decision only — not a trigger (see [`drift.md`](drift.md)). On CI, the daily job skips the routine train so this gate evaluates the **frozen** `models/baseline_latest` against new data; if it fires, the new baseline is **committed to git** for the next runs.

```bash
.venv/bin/python -m ml.retrain_check --dry-run -v
.venv/bin/python -m ml.retrain_check -v
.venv/bin/python -m ml.retrain_check --run-drift -v
.venv/bin/python -m ml.retrain_check --mae-ratio 1.5 --min-reference 50 --no-mlflow -v
```

Decision → `reports/retrain_latest/decision.json` (`should_retrain`, `trigger_reason`, `mae_ratio`, …).

## Dashboard (RF-10)

**Local (full RF-10):** Streamlit — try-predict + monitoring + good deals.

```bash
export RENT_API_URL=http://127.0.0.1:8000   # optional; else local model file
.venv/bin/streamlit run dashboard/app.py
```

**Public:** second Render Web Service (Streamlit) → API — [`doc/render.md`](render.md) §2 (`RENT_API_URL`).

Good deals / monitoring need local `features_latest`, `baseline_latest`, and `reports/` (not on the public UI service).

## Serving API

Requires a trained model at `models/baseline_latest/model.joblib` (from `ml.train`).

```bash
.venv/bin/uvicorn api.main:app --reload --port 8000
```

| Endpoint | Role |
|----------|------|
| `GET /health` | Model load status |
| `POST /predict` | Fair €/m² prediction; optional `price_per_m2_monthly` → `gap_pct` + `deal_label` |
| `GET /docs` | OpenAPI UI |

Example:

```bash
curl -s http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{
  "surface_m2": 80,
  "rooms": 3,
  "distance_from_center_km": 2.5,
  "area_price_per_m2_hist": 30.0,
  "publication_month": 9,
  "municipio": "I",
  "price_per_m2_monthly": 25.0
}'
```

Deal labels use a ±10% band on `(actual - predicted) / predicted`: `good_deal`, `fair_price`, `above_market`.

## Docker (API)

Build from the repo root (includes `models/` from the build context if you already trained):

```bash
docker build -t rent-tracker-api .
docker run --rm -p 8000:8000 rent-tracker-api
```

If the image has no model, mount a local `models/` (needs `baseline_latest/model.joblib`):

```bash
docker run --rm -p 8000:8000 -v "$PWD/models:/app/models:ro" rent-tracker-api
```

Then `GET http://127.0.0.1:8000/health` and `POST /predict` as above.

Cloud deploy steps: [`doc/render.md`](render.md).

## Tests

```bash
.venv/bin/python -m pytest -q
```
