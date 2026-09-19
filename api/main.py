"""FastAPI app: fair rent prediction (RF-06) + deal label (RF-07) + admin OMI ingest."""

from __future__ import annotations

import json

import hmac
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile

from api.ingest import IngestError, ingest_omi_csv, ingest_token
from api.predictor import DEFAULT_MODEL_PATH, ModelPredictor
from api.profile_history import load_profile_series
from api.schemas import (
    HealthResponse,
    IngestResponse,
    PredictRequest,
    PredictResponse,
    ProfileHistoryResponse,
    ProfileHistoryPoint,
    ProfileNextPrediction,
    TipologieResponse,
    ZonesResponse,
    SightingCreate,
    SightingResponse,
)
from api.tipologie import list_tipologie
from api.zone import list_zones
from ml.train import DEFAULT_INPUT


import uuid
from datetime import datetime, timezone
from api.sightings import append_sighting, DEFAULT_PATH

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

from api.cloud_store import ensure_features_latest


# function run by uvicorn to create the app
def create_app(
    model_path: Path | None = None,
    features_path: Path | None = None,
    sightings_path: Path | None = None,
) -> FastAPI:
    path = model_path or DEFAULT_MODEL_PATH
    feats = features_path if features_path is not None else DEFAULT_INPUT
    sightings = sightings_path if sightings_path is not None else DEFAULT_PATH
    # lifespane (HOOKS) must be defining startup + shutdown ()
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        global _predictor
        try:
            # Pull features from R2/DVC when missing (Render Docker has no gitignored JSONL).
            if not Path(feats).is_file():
                ensure_features_latest(feats)


            # startup
            _predictor = ModelPredictor(path, features_path=feats)
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
            "GET /profile/history for semester series + next-semester forecast. "
            "GET /meta/tipologie for distinct tipologias in features. "
            "GET /meta/zones for zona_omi + OMI description labels. "
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

    @app.get("/meta/tipologie", response_model=TipologieResponse)
    def meta_tipologie() -> TipologieResponse:
        path = Path(feats)
        if _predictor is not None and _predictor.features_path is not None:
            path = Path(_predictor.features_path)
        return TipologieResponse(tipologie=list_tipologie(path))

    @app.get("/meta/zones", response_model=ZonesResponse)
    def meta_zones() -> ZonesResponse:
        path = Path(feats)
        if _predictor is not None and _predictor.features_path is not None:
            path = Path(_predictor.features_path)
        return ZonesResponse(zones=list_zones(path))

    @app.post("/predict", response_model=PredictResponse)
    def predict(body: PredictRequest) -> PredictResponse:
        predictor = get_predictor()
        features = body.model_dump(exclude={"price_per_m2_monthly"})
        result = predictor.score(features, actual_price_per_m2=body.price_per_m2_monthly)
        return PredictResponse(**result)

    @app.post("/sightings", response_model=SightingResponse)
    def create_sighting(body: SightingCreate) -> SightingResponse:
        predictor = get_predictor()
        features = {
            "zona_omi": body.zona_omi.strip(),
            "tipologia": body.tipologia,
            "stato": body.stato,
        }
        features_file = predictor.features_path

        if features_file is None or not Path(features_file).is_file():
            raise HTTPException(
                status_code=503,
                detail="features_latest.jsonl not available on API (dvc pull / mount)",
            )
        
        series = load_profile_series(
            features_file,
            zona_omi=body.zona_omi,
            tipologia=body.tipologia,
            stato=body.stato,
        )

        if not series:
            raise HTTPException(status_code=404, detail="No OMI history for this zona_omi / tipologia / stato")

        loc_mid_lag = float(series[-1]["price_per_m2_monthly"])  # last mid → lag next semester
        features["loc_mid_lag"] = loc_mid_lag
        result = predictor.score(features, actual_price_per_m2=body.asking_eur_m2)
        record = {
            "sighting_id": str(uuid.uuid4()),
            "submitted_at": str(datetime.now(timezone.utc)),
            "source": "user",
            "zona_omi": body.zona_omi.strip(),
            "tipologia": body.tipologia,
            "stato": body.stato,
            "asking_eur_m2": body.asking_eur_m2,
            "predicted_price_per_m2_monthly": result["predicted_price_per_m2_monthly"],
            "gap_pct": result["gap_pct"],
            "deal_label": result["deal_label"],
            "deal_basis": result["deal_basis"],
        }

        (status, duplicate_of) = append_sighting(sightings, record)
        
        return SightingResponse(**record, status=status, duplicate_of=duplicate_of)

    @app.get("/profile/history", response_model=ProfileHistoryResponse)
    def profile_history(
        zona_omi: str,
        tipologia: str,
        stato: str,
    ) -> ProfileHistoryResponse:
        predictor = get_predictor()
        features_file = predictor.features_path
        if features_file is None or not Path(features_file).is_file():
            raise HTTPException(
                status_code=503,
                detail="features_latest.jsonl not available on API (dvc pull / mount)",
            )
        series = load_profile_series(
            Path(features_file),
            zona_omi=zona_omi,
            tipologia=tipologia,
            stato=stato,
        )
        if not series:
            raise HTTPException(
                status_code=404,
                detail="No OMI history for this zona_omi / tipologia / stato",
            )
        last = series[-1]
        loc_mid_lag = float(last["price_per_m2_monthly"])
        scored = predictor.score(
            {
                "zona_omi": zona_omi.strip(),
                "tipologia": tipologia,
                "stato": stato,
                "loc_mid_lag": loc_mid_lag,
            },
            include_shap=False,
        )
        return ProfileHistoryResponse(
            zona_omi=zona_omi.strip(),
            tipologia=tipologia,
            stato=stato,
            test_semester=str(last["semester"]),
            series=[ProfileHistoryPoint(**p) for p in series],
            next_prediction=ProfileNextPrediction(
                predicted_price_per_m2_monthly=scored["predicted_price_per_m2_monthly"],
                loc_mid_lag=loc_mid_lag,
                omi_loc_min=scored.get("omi_loc_min"),
                omi_loc_max=scored.get("omi_loc_max"),
                omi_half_width=scored.get("omi_half_width"),
                band_source=scored.get("band_source"),
            ),
        )

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