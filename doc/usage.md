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

## Tests

```bash
.venv/bin/python -m pytest -q
```
