"""Request/response models for the serving API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    zona_omi: str = Field(..., min_length=1)
    tipologia: str = Field(..., min_length=1)
    stato: str = Field(..., min_length=1)
    publication_month: int = Field(..., ge=1, le=12)
    loc_mid_lag: float | None = None
    # Optional asking €/m² the user observed (ad); RF-07 classify vs model fair
    price_per_m2_monthly: float | None = Field(
        default=None,
        gt=0,
        description="Asking rent €/m² from a listing the user saw (not OMI mid).",
    )


class PredictResponse(BaseModel):
    predicted_price_per_m2_monthly: float
    features_used: list[str]
    model_path: str
    price_per_m2_monthly: float | None = None
    gap_pct: float | None = None
    deal_label: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str | None = None
