# Serve FastAPI predict API (RF-06 / Phase 3).
FROM python:3.12-slim

WORKDIR /app

# OpenMP runtime used by scikit-learn HistGradientBoosting
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Serving imports FEATURE_COLS from ml.train
COPY api/ api/
COPY ml/ ml/
# Bake local baseline if present in build context 
COPY models/ models/

# for debugging purposes
ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Build: (for building the image)
#   docker build -t rent-tracker-api .
# Run: (for running the container)
#   docker run --rm -p 8001:8000 rent-tracker-api