# Public try-predict UI → Render API (option A)

Render keeps serving the model; this UI only calls `/predict`.

| Surface | Role |
|---------|------|
| Render | FastAPI `/predict`, `/health` — [`render.md`](render.md) |
| HF Space (Gradio) | Form → HTTP to Render via `RENT_API_URL` |
| Local Streamlit | Full RF-10 (`dashboard/app.py`: monitoring + good deals) |

Secret / variable on the Space:

| Name | Value |
|------|--------|
| `RENT_API_URL` | `https://<your-service>.onrender.com` |

## Hugging Face — use Gradio (free SDK)

Native Streamlit SDK is gone; **Docker** is often marked Paid. If **Gradio** shows as free in the create form, use that.

1. Confirm API:
   ```bash
   curl -s https://<your-service>.onrender.com/health
   ```
2. [Create new Space](https://huggingface.co/new-space)
   - SDK: **Gradio**
   - Hardware: free CPU / whatever the form offers without PRO
3. Connect GitHub `SimBoex/Rent-tracker` **or** push this repo into the Space.
4. Space settings → **Variables and secrets** → `RENT_API_URL` = Render URL.
5. Point the app file at Gradio (README YAML on the Space, or settings):

```yaml
---
title: Roma Rent Monitor
sdk: gradio
app_file: dashboard/gradio_app.py
---
```

6. Optional: if the build OOMs on full `requirements.txt`, replace Space deps with [`requirements-space.txt`](../requirements-space.txt) (rename/copy to `requirements.txt` on the Space, or install only those packages).

7. Open the Space → Submit → check predicted €/m².

## Local Gradio check

```bash
.venv/bin/pip install -r requirements-space.txt
# terminal 1
.venv/bin/uvicorn api.main:app --port 8000
# terminal 2
export RENT_API_URL=http://127.0.0.1:8000
.venv/bin/python dashboard/gradio_app.py
```

## Local Streamlit (full dashboard)

```bash
export RENT_API_URL=http://127.0.0.1:8000   # optional
.venv/bin/streamlit run dashboard/app.py
```

## Notes

- Do not put R2/AWS keys on the Space (RNF-03).
- Free Render sleeps: first call after idle can be slow — retry once.
- If Gradio also becomes Paid on your account, fall back to Streamlit Community Cloud with `dashboard/app.py` (same `RENT_API_URL` secret).
