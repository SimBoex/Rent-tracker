# Rent-tracker

Scrapes Rome rental listings, cleans/features them, and trains a baseline model for fair rent (€/m²/month), with local MLflow tracking.

Full requirements: [`SOR-roma-rent-monitor.md`](SOR-roma-rent-monitor.md).  
Detailed commands & MLflow UI: [`doc/usage.md`](doc/usage.md).  
Evidently drift: [`doc/drift.md`](doc/drift.md).  
Render deploy: [`doc/render.md`](doc/render.md).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

From the repo root:

```bash
.venv/bin/python run_pipeline.py --skip-scrape -v          # clean → features → train
.venv/bin/python run_pipeline.py --max-pages 5 -v           # scrape + full pipeline
.venv/bin/uvicorn api.main:app --reload --port 8000        # predict API (needs models/baseline_latest)
.venv/bin/python -m ml.drift_report -v                     # Evidently drift HTML → reports/
.venv/bin/python -m ml.retrain_check --dry-run -v          # RF-09 MAE gate (no train)
.venv/bin/streamlit run dashboard/app.py                   # RF-10 monitoring + good deals
docker build -t rent-tracker-api . && docker run --rm -p 8000:8000 rent-tracker-api
.venv/bin/python -m pytest -q
```

Outputs land under `data/processed/`, `models/`, `reports/`, and `mlflow.db` (gitignored where appropriate).
