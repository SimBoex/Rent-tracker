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

## Official stage commands

```bash
.venv/bin/python etl/extract/immobiliare_scraper.py --max-pages 5 -v
.venv/bin/python -m etl.transform.clean_phase -v
.venv/bin/python -m etl.transform.features_phase -v
.venv/bin/python -m ml.train -v
```

Train only:

```bash
.venv/bin/python -m ml.train --input data/processed/features_latest.jsonl -v
.venv/bin/python -m ml.train --no-mlflow -v
```

## Outputs

| Stage | Path |
|-------|------|
| Raw scrape | `data/raw/immobiliare_roma_<ts>/` |
| Clean + quality | `data/processed/listings_*.jsonl`, `rejected_*.jsonl`, `quality_*.json` |
| Features | `data/processed/features_*.jsonl` (+ `features_latest.jsonl`) |
| Model | `models/baseline_<ts>/`, `models/baseline_latest/` |
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

## Tests

```bash
.venv/bin/python -m pytest -q
```
