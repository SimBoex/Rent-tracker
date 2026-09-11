"""FastAPI app: fair rent prediction (RF-06) + deal label (RF-07)."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from api.predictor import DEFAULT_MODEL_PATH, ModelPredictor
from api.schemas import HealthResponse, PredictRequest, PredictResponse

_predictor: ModelPredictor | None = None


def get_predictor() -> ModelPredictor:
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _predictor

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
            "Predict fair rent €/m²/month from OMI zone features "
            "(«Agenzia Entrate – OMI») and classify vs model."
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

    return app


app = create_app()
