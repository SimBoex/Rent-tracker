# Render deployment

Two free Web Services from the same GitHub repo:

| Service | Role |
|---------|------|
| API (Docker) | FastAPI `/predict`, `/profile/history`, `/health` |
| UI (Python) | Streamlit `dashboard/app.py` → calls the API via `RENT_API_URL` |

## 1. API service (model)

1. Push the repository to GitHub.
2. Render → `New +` → `Web Service` → connect `Rent-tracker`.
3. Runtime: **Docker**.
4. Root directory: repo root.
5. Health check path: `/health`.
6. Create and wait for deploy.

```bash
curl -s https://<api-service>.onrender.com/health
```

Notes:
- Container exposes `/health`, `/predict`, `/profile/history`, `/docs`.
- Image copies `api/`, `ml/`, `etl/` (needed for `semester_key` / band lookup) and `models/`.
- `models/baseline_latest/model.joblib` must be in the image or `/health` is `degraded` and `/predict` returns `503`.
- `GET /profile/history` needs `data/processed/features_latest.jsonl`. The Docker image ships the DVC pointer (`.dvc`); at startup the API downloads the JSONL from the DVC remote on R2 when `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_ENDPOINT_URL` are set (same as ingest). Without those env vars, profile history returns `503`.
- If the port is wrong, make the Docker CMD read `PORT`.

## 2. UI service (Streamlit)

Same repo, **second** Web Service (no Docker).

1. Render → `New +` → `Web Service` → same repo `SimBoex/Rent-tracker`.
2. Settings:

| Field | Value |
|-------|--------|
| Name | e.g. `rent-tracker-ui` |
| Language | **Python 3** |
| Branch | `main` |
| Root Directory | *(leave empty)* |
| Build Command | `pip install -r requirements-space.txt` |
| Start Command | `python -m streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true` |
| Instance | Free |

3. **Environment** → add:

| Key | Value |
|-----|--------|
| `RENT_API_URL` | `https://<api-service>.onrender.com` |
| `MONITORING_URL` | *(optional)* raw `reports/monitoring_latest.json` on `main` |

(Use the **API** service URL from step 1, no trailing slash.)

4. Deploy → open `https://<ui-service>.onrender.com` → **Profile history** + **Monitoring** + admin upload expander.

The tipologia selectbox calls `GET /meta/tipologie` on the API (features live on the API host / R2 pull). Without features on the API the list is empty and the UI shows an error. Locally (no `RENT_API_URL`) it reads `features_latest.jsonl` if present. Profile history can **add multiple profiles** and overlay semester mid lines on one chart.

## 2b. Cloud OMI upload (Render → R2 → GitHub Actions)

Durable path (no Render disk needed for CSV persistence):

```text
UI upload → API /ingest/omi → R2 (omi-ingest/) → workflow_dispatch omi-monitoring
         → CI: dvc pull + pull_ingest_inbox → features → drift → retrain
```

### API service env (Render)

| Key | Value |
|-----|--------|
| `INGEST_TOKEN` | long random secret (type the same in the UI) |
| `AWS_ACCESS_KEY_ID` | same R2/S3 key as GitHub Actions / DVC |
| `AWS_SECRET_ACCESS_KEY` | same secret |
| `AWS_ENDPOINT_URL` | R2 endpoint (`https://<ACCOUNT_ID>.r2.cloudflarestorage.com`) |
| `INGEST_S3_BUCKET` | optional; default `rent-tracker-data` |
| `GITHUB_TOKEN` | PAT with `actions:write` (and `contents:read`) on this repo |
| `GITHUB_REPOSITORY` | `SimBoex/Rent-tracker` |

### `INGEST_TOKEN` on Render (UI online)

This is **not** an OMI / Agenzia / GitHub token. You invent a shared admin password so only you can `POST /ingest/omi`.

1. **Generate** (once, on your machine):
   ```bash
   openssl rand -hex 32
   ```
   Copy the output.
2. **Set it on the API service only** (not on the Streamlit UI):
   - Render Dashboard → open the **API** (Docker) service  
   - **Environment** → **Add Environment Variable**  
   - Key: `INGEST_TOKEN` · Value: the string from step 1  
   - Save (API redeploys)
3. **UI env** still only needs `RENT_API_URL` (and optional snapshot URLs). Do **not** put `INGEST_TOKEN` on the UI service.
4. **Use it in the live UI**:
   - Open `https://<ui-service>.onrender.com`
   - Scroll to expander **Admin · upload OMI semester CSV**
   - Field **Ingest token** → paste the **same** string as the API env
   - Choose an OMI `*VALORI*.csv` → **Upload semester**

Mismatch → `401`. Missing on API → `503` (ingest disabled). You re-type the token in the UI each upload; it is not stored in the dashboard.

Notes:
- Without `INGEST_TOKEN` → `/ingest/omi` returns 503.
- With AWS keys → upload + SHA-256 dedupe against **remote** manifest `omi-ingest/manifest.json`.
- With `GITHUB_TOKEN` + checkbox “run monitoring” → dispatches `.github/workflows/daily_monitoring.yml`.
- CI step `pull_ingest_inbox` copies new CSVs into `data/raw/omi/` before the pipeline.

### GitHub Actions secrets (already used by DVC)

Keep `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL`. Optional `INGEST_S3_BUCKET` if not default.

Public snapshots (committed by OMI CI):
- `reports/monitoring_latest.json` — aggregate drift / retrain metrics only (no raw paths)  
- `reports/good_deals_latest.json` — legacy zone-label snapshot (UI no longer uses it)

Profile history (€/m² series + next-semester forecast) comes from the **API** (`GET /profile/history`). On Render the API pulls `features_latest.jsonl` from R2 (DVC cache) at startup when AWS credentials are set.

Always attribute «Agenzia Entrate – OMI» (UI footer + snapshot metadata).

Until the first successful export, Monitoring says the snapshot is not ready yet.

Predict body uses OMI fields: `zona_omi`, `tipologia`, `stato`, optional `loc_mid_lag`; optional `price_per_m2_monthly` = user asking €/m². See [`usage.md`](usage.md).

## 3. After training a new OMI model locally

Render’s Docker API only sees what is **in git** at build time (`models/baseline_latest/`).

1. Commit & push `models/baseline_latest/` (+ code if changed).  
2. Render → API service → **Manual Deploy**.  
3. `curl -s https://<api>/health` → `model_loaded: true`.  
4. UI: confirm `RENT_API_URL`; redeploy UI only if Streamlit code/env changed.  
5. Optional: run *omi-monitoring* so `reports/monitoring_latest.json` on `main` refresh Monitoring.

Raw OMI CSVs stay out of git — use DVC ([`dvc.md`](dvc.md)). Checklist also in [`omi.md`](omi.md) / [`usage.md`](usage.md).

First request after idle can be slow (both free services sleep). Wake the API with `/health`, then retry Profile history.

## Local check

```bash
.venv/bin/uvicorn api.main:app --port 8000
export RENT_API_URL=http://127.0.0.1:8000
.venv/bin/streamlit run dashboard/app.py
```
