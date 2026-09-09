# Rent-tracker

Scrapes Rome rental listings, cleans/features them, and trains a baseline model for fair rent (€/m²/month), with local MLflow tracking.

Full requirements: [`SOR-roma-rent-monitor.md`](SOR-roma-rent-monitor.md).  
Detailed commands & MLflow UI: [`doc/usage.md`](doc/usage.md).

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
.venv/bin/python -m pytest -q
```

Outputs land under `data/processed/`, `models/`, and `mlflow.db` (gitignored where appropriate).
