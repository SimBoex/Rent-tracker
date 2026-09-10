# Render deployment

Two free Web Services from the same GitHub repo:

| Service | Role |
|---------|------|
| API (Docker) | FastAPI `/predict`, `/health` |
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
- Container exposes `/health`, `/predict`, `/docs`.
- `models/baseline_latest/model.joblib` must be in the image or `/health` is `degraded` and `/predict` returns `503`.
- If the port is wrong, make the Docker CMD read `PORT`.

## 2. UI service (Streamlit try-predict)

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

(Use the **API** service URL from step 1, no trailing slash.)

4. Deploy → open `https://<ui-service>.onrender.com` → **Predict**.

First request after idle can be slow (both free services sleep). Wake the API with `/health`, then retry Predict.

## Local check

```bash
.venv/bin/uvicorn api.main:app --port 8000
export RENT_API_URL=http://127.0.0.1:8000
.venv/bin/streamlit run dashboard/app.py
```
