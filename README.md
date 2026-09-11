# Rent-tracker

Ingests Rome **OMI** locazione quotazioni (Agenzia delle Entrate), builds features, and trains a baseline model for fair rent (€/m²/month), with local MLflow tracking. Serves predict API + Streamlit UI on Render.

Requirements: [`SOR-roma-rent-monitor.md`](SOR-roma-rent-monitor.md) (v1 historical) · [`SOR2-roma-rent-monitor.md`](SOR2-roma-rent-monitor.md) (current / OMI).  
OMI download & columns: [`doc/omi.md`](doc/omi.md).  
Commands: [`doc/usage.md`](doc/usage.md).  
Evidently drift: [`doc/drift.md`](doc/drift.md).  
Render (API + UI): [`doc/render.md`](doc/render.md).  
DVC: [`doc/dvc.md`](doc/dvc.md).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

Download OMI CSVs into `data/raw/omi/` (see `doc/omi.md`):

```bash
.venv/bin/python run_pipeline.py -v                             # real CSVs → features → train
.venv/bin/uvicorn api.main:app --reload --port 8000             # predict API
.venv/bin/python -m ml.drift_report -v
.venv/bin/python -m ml.retrain_check --dry-run -v
.venv/bin/streamlit run dashboard/app.py
.venv/bin/python -m pytest -q
docker build -t rent-tracker-api . && docker run --rm -p 8000:8000 rent-tracker-api
```

CI: `.github/workflows/ci.yml` (pytest) + `daily_monitoring.yml` (`omi-monitoring`: DVC → OMI load → drift → retrain).  
Cloud cutover checklist: [`doc/omi.md`](doc/omi.md) § Sync to cloud · [`doc/render.md`](doc/render.md) · [`doc/dvc.md`](doc/dvc.md).  
Outputs: `data/processed/`, `models/`, `reports/`, `mlflow.db`.
