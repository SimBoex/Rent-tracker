# Serve FastAPI predict API (RF-06 / Phase 3).
FROM python:3.12-slim

WORKDIR /app

# OpenMP runtime used by scikit-learn HistGradientBoosting
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Serving: api → ml.train / omi_band / profile_history → etl.semester
COPY api/ api/
COPY ml/ ml/
COPY etl/ etl/
# Bake local baseline if present in build context
COPY models/ models/

# Profile history + OMI band need features_latest on the API host.
# File is DVC-tracked (gitignored): run `dvc pull` before `docker build`, then:
#   COPY data/processed/features_latest.jsonl data/processed/features_latest.jsonl
RUN mkdir -p data/processed

# for debugging purposes
ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Build:
#   docker build -t rent-tracker-api .
# Run:
#   docker run --rm -p 8001:8000 rent-tracker-api
# With features (local):
#   docker run --rm -p 8001:8000 \
#     -v "$PWD/data/processed/features_latest.jsonl:/app/data/processed/features_latest.jsonl:ro" \
#     rent-tracker-api
