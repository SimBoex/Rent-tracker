"""Request/response models for the serving API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    surface_m2: float = Field(..., gt=0)
    rooms: float = Field(..., gt=0)
    distance_from_center_km: float | None = None
    area_price_per_m2_hist: float | None = None
    publication_month: int = Field(..., ge=1, le=12)
    municipio: str = Field(..., min_length=1)
    # Optional actual €/m² for RF-07 deal classification
    price_per_m2_monthly: float | None = Field(default=None, gt=0)


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
