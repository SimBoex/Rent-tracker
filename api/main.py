"""FastAPI app: fair rent prediction (RF-06) + deal label (RF-07) + admin OMI ingest."""

from __future__ import annotations

import hmac
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile

from api.ingest import IngestError, ingest_omi_csv, ingest_token
from api.predictor import DEFAULT_MODEL_PATH, ModelPredictor
from api.schemas import (
    HealthResponse,
    IngestResponse,
    PredictRequest,
    PredictResponse,
)

_predictor: ModelPredictor | None = None


def get_predictor() -> ModelPredictor:
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _predictor


def _require_ingest_token(x_ingest_token: str | None) -> None:
    expected = ingest_token()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Ingest disabled: set INGEST_TOKEN on the API service",
        )
    provided = (x_ingest_token or "").strip()
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Ingest-Token")


# function run by uvicorn to create the app
def create_app(model_path: Path | None = None) -> FastAPI:
    path = model_path or DEFAULT_MODEL_PATH

    # lifespane (HOOKS) must be defining startup + shutdown ()
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        global _predictor
        try:
            # startup
            _predictor = ModelPredictor(path)
        except FileNotFoundError as exc:
            _predictor = None
            app.state.model_load_error = str(exc)
        else:
            app.state.model_load_error = None
        yield
        # shutdown
        _predictor = None

    app = FastAPI(
        title="Roma Rent Monitor API",
        description=(
            "OMI-trained fair rent €/m²/month from zone features "
            "(«Agenzia Entrate – OMI»). Optional asking €/m² → deal label vs fair. "
            "Admin: POST /ingest/omi (X-Ingest-Token)."
        ),
        version="0.2.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        loaded = _predictor is not None
        return HealthResponse(
            status="ok" if loaded else "degraded",
            model_loaded=loaded,
            model_path=str(_predictor.model_path) if _predictor else None,
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(body: PredictRequest) -> PredictResponse:
        predictor = get_predictor()
        features = body.model_dump(exclude={"price_per_m2_monthly"})
        result = predictor.score(features, actual_price_per_m2=body.price_per_m2_monthly)
        return PredictResponse(**result)

    @app.post("/ingest/omi", response_model=IngestResponse)
    async def ingest_omi(
        file: UploadFile = File(...),
        run_pipeline: bool = Form(False),
        x_ingest_token: str | None = Header(default=None, alias="X-Ingest-Token"),
    ) -> IngestResponse:
        """Store a new OMI VALORI CSV (dedupe by SHA-256). Requires INGEST_TOKEN."""
        _require_ingest_token(x_ingest_token)
        raw = await file.read()
        try:
            result = ingest_omi_csv(
                filename=file.filename or "upload.csv",
                content=raw,
                run_pipeline=run_pipeline,
            )
        except IngestError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

        if result.status == "duplicate":
            return IngestResponse(
                status="duplicate",
                sha256=result.sha256,
                duplicate_of=result.duplicate_of,
                detail="Identical file already present (content SHA-256 match)",
            )
        return IngestResponse(
            status=result.status,
            sha256=result.sha256,
            saved_as=result.saved_as,
            n_rows=result.n_rows,
            semester=result.semester,
            pipeline=result.pipeline,
            cloud_key=result.cloud_key,
            workflow=result.workflow,
        )

    return app


app = create_app()