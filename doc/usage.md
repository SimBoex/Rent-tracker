# Usage details

Always run commands from the **repo root**.

## Pipeline orchestrator

`run_pipeline.py` runs: **OMI load → features → train (optional)**.  
Put CSVs in `data/raw/omi/` first (see [`omi.md`](omi.md)).

```bash
.venv/bin/python run_pipeline.py [options]
.venv/bin/python run_pipeline.py -h
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--skip-train` | off | Stop after features |
| `--no-mlflow` | off | Skip MLflow logging (still writes `models/`) |
| `-v` | off | Verbose logs |

Examples:

```bash
.venv/bin/python run_pipeline.py -v
.venv/bin/python run_pipeline.py --skip-train -v
.venv/bin/python run_pipeline.py --no-mlflow -v
```

## CI cadence (GitHub Actions)

Workflow [`.github/workflows/daily_monitoring.yml`](../.github/workflows/daily_monitoring.yml) (`omi-monitoring`):

- **Manual:** Actions → *omi-monitoring* → *Run workflow* (after you `dvc push` new CSVs)
- **Schedule:** quarterly nudge (still needs CSVs in DVC)
- **Steps:** `dvc pull` → require `*VALORI*.csv` → `run_pipeline.py --skip-train` → DVC push → `ml.drift_report` → `ml.retrain_check` → dashboard snapshots
- **Artifacts:** drift / retrain / snapshots / metrics (14 days)

Unit tests on push/PR: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).

## Realign local → cloud

After new OMI downloads (or code changes):

1. **Data (private):** put `*VALORI*.csv` in `data/raw/omi/` (≥2–3 semesters preferred; provincia RM OK — loader keeps only comune Roma).  
2. **Train locally (optional but recommended before Render):**  
   `.venv/bin/python run_pipeline.py -v`  
3. **DVC:**  
   `dvc add data/raw`  
   `dvc add data/processed/features_latest.jsonl`  
   `dvc push`  
   commit the updated `*.dvc` pointers (not the CSVs).  
4. **Git:** commit & push code + `models/baseline_latest/` (Docker API bakes this).  
5. **Render:** Manual Deploy on the **API** service (and UI if `dashboard/` changed). Check `/health` then `/docs` with OMI fields.  
6. **CI (optional):** Actions → *omi-monitoring* → Run workflow (pulls DVC, drift, snapshots).

Full detail: [`omi.md`](omi.md), [`dvc.md`](dvc.md), [`render.md`](render.md).

## Stage commands

```bash
.venv/bin/python -m etl.extract.omi_loader -v
.venv/bin/python -m etl.transform.omi_features -v
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

Each train run fingerprints the input JSONL (SHA-256 + size + row count) and writes `dataset.json` next to the model. The same block is embedded in `metrics.json` under `dataset`, and MLflow logs `dataset_sha256` + the artifact. Raw OMI CSVs stay gitignored; sync via [`dvc.md`](dvc.md).

```bash
cat models/baseline_latest/dataset.json
```

## Outputs

| Stage | Path |
|-------|------|
| Raw OMI CSV | `data/raw/omi/*.csv` |
| Raw OMI JSONL | `data/raw/omi/omi_quotazioni_latest.jsonl` |
| Features | `data/processed/features_*.jsonl` (+ `features_latest.jsonl`) |
| Model | `models/baseline_<ts>/`, `models/baseline_latest/` |
| Drift | `reports/drift_<ts>/`, `reports/drift_latest/` |
| Retrain gate | `reports/retrain_<ts>/`, `reports/retrain_latest/` |
| Dashboard | `streamlit run dashboard/app.py` |
| MLflow | `mlflow.db` (SQLite) |

## MLflow UI

Training uses SQLite (`mlflow.db`). Prefer this over `./mlruns` with MLflow 3.

```bash
.venv/bin/mlflow ui --backend-store-uri sqlite:///$(pwd)/mlflow.db --port 5000
```

Open http://127.0.0.1:5000 — experiment `roma-rent-baseline`.

## Predict API

```bash
.venv/bin/uvicorn api.main:app --reload --port 8000
```

| Endpoint | Role |
|----------|------|
| `GET /health` | Model load status |
| `POST /predict` | Fair €/m²; optional `price_per_m2_monthly` = **asking €/m² you observed** → `gap_pct` + `deal_label` |
| `GET /docs` | OpenAPI UI |

Example:

```bash
curl -s http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{
  "zona_omi": "B12",
  "tipologia": "Abitazioni civili",
  "stato": "NORMALE",
  "publication_month": 6,
  "loc_mid_lag": 20.0,
  "price_per_m2_monthly": 18.0
}'
```

Deal labels (±10% on `(asking - fair) / fair`): `below_omi_band`, `in_band`, `above_omi_band`.  
Optional `price_per_m2_monthly` is the user’s asking rent÷m² (portal ad), not an OMI mid lookup.

## Docker (API)

```bash
docker build -t rent-tracker-api .
docker run --rm -p 8000:8000 rent-tracker-api
```

If the image has no model, mount local `models/`:

```bash
docker run --rm -p 8000:8000 -v "$PWD/models:/app/models:ro" rent-tracker-api
```

Cloud: [`render.md`](render.md).

## Tests

```bash
.venv/bin/python -m pytest -q
```
