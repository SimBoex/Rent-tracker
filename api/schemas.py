"""Request/response models for the serving API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    zona_omi: str = Field(..., min_length=1)
    tipologia: str = Field(..., min_length=1)
    stato: str = Field(..., min_length=1)
    loc_mid_lag: float | None = None
    # Optional asking €/m² the user observed (ad); RF-07 classify vs OMI band or model fair
    price_per_m2_monthly: float | None = Field(
        default=None,
        gt=0,
        description="Asking rent €/m² from a listing the user saw (not OMI mid).",
    )


class ShapContribution(BaseModel):
    feature: str
    shap_value: float


class PredictResponse(BaseModel):
    predicted_price_per_m2_monthly: float
    features_used: list[str]
    model_path: str
    # OMI locazione band for the zone segment (latest semester in features); not model CI
    omi_loc_min: float | None = None
    omi_loc_max: float | None = None
    omi_half_width: float | None = None
    band_source: str | None = None
    price_per_m2_monthly: float | None = None
    gap_pct: float | None = None
    deal_label: str | None = None
    # omi_band = asking vs OMI loc min/max; model_pct = ±10% vs model fair
    deal_basis: str | None = None
    # RF-10d: TreeSHAP contributions (preprocessed feature space)
    shap_values: list[ShapContribution] | None = None
    shap_base_value: float | None = None


class ProfileHistoryPoint(BaseModel):
    semester: str
    price_per_m2_monthly: float
    role: str  # train | test (last OMI semester = test)
    omi_loc_min: float | None = None
    omi_loc_max: float | None = None
    loc_mid_lag: float | None = None


class ProfileNextPrediction(BaseModel):
    predicted_price_per_m2_monthly: float
    loc_mid_lag: float
    omi_loc_min: float | None = None
    omi_loc_max: float | None = None
    omi_half_width: float | None = None
    band_source: str | None = None


class ProfileHistoryResponse(BaseModel):
    zona_omi: str
    tipologia: str
    stato: str
    test_semester: str
    series: list[ProfileHistoryPoint]
    next_prediction: ProfileNextPrediction
    source_attribution: str = "Agenzia Entrate – OMI"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str | None = None


class TipologieResponse(BaseModel):
    tipologie: list[str]


class ZoneItem(BaseModel):
    zona_omi: str
    descr: str | None = None
    label: str


class ZonesResponse(BaseModel):
    zones: list[ZoneItem]


class IngestResponse(BaseModel):
    status: str
    sha256: str
    saved_as: str | None = None
    n_rows: int | None = None
    semester: str | None = None
    duplicate_of: str | None = None
    pipeline: str | None = None
    cloud_key: str | None = None
    workflow: str | None = None
    detail: str | None = None

#####
from datetime import datetime

class SightingCreate(BaseModel):
    zona_omi: str
    tipologia: str
    stato: str
    asking_eur_m2: float

class SightingStored(BaseModel):

    sighting_id: str
    submitted_at: datetime
    source: str

    zona_omi: str
    tipologia: str
    stato: str
    asking_eur_m2: float
    predicted_price_per_m2_monthly: float
    gap_pct: float
    deal_label: str
    deal_basis: str

class SightingResponse(SightingStored):
    status: str
    duplicate_of: str | None = None
####