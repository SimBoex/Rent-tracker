"""Load baseline model and score OMI zone rows (RF-06 / RF-07)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from api.omi_band import band_payload, build_band_index
from ml.train import DEFAULT_INPUT, FEATURE_COLS, MODELS_DIR

DEFAULT_MODEL_PATH = MODELS_DIR / "baseline_latest" / "model.joblib"

# Relative gap vs predicted fair €/m²: |actual - pred| / pred
DEAL_BAND = 0.10

BELOW_OMI_BAND = "below_omi_band"
IN_BAND = "in_band"
ABOVE_OMI_BAND = "above_omi_band"
# Back-compat alias used by dashboard filters
GOOD_DEAL = BELOW_OMI_BAND


class ModelPredictor:
    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        features_path: Path | None = DEFAULT_INPUT,
    ):
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
        loaded = joblib.load(model_path)
        if not isinstance(loaded, Pipeline):
            raise TypeError(f"Expected sklearn Pipeline, got {type(loaded)!r}")
        self.pipeline: Pipeline = loaded
        self.model_path = model_path
        self.features_path = features_path
        self._band_index = (
            build_band_index(features_path) if features_path is not None else {}
        )

    def predict_price_per_m2(self, features: dict[str, Any]) -> float:
        row = {c: features.get(c) for c in FEATURE_COLS}
        X = pd.DataFrame([row], columns=FEATURE_COLS)
        return float(self.pipeline.predict(X)[0])

    @staticmethod
    def classify_deal(actual: float, predicted: float, band: float = DEAL_BAND) -> str:
        if predicted <= 0:
            raise ValueError("predicted must be > 0")
        gap = (actual - predicted) / predicted
        if gap <= -band:
            return BELOW_OMI_BAND
        if gap >= band:
            return ABOVE_OMI_BAND
        return IN_BAND

    def _resolve_omi_band(self, features: dict[str, Any]) -> dict[str, Any] | None:
        lo, hi = features.get("omi_loc_min"), features.get("omi_loc_max")
        if lo is not None and hi is not None:
            return band_payload(float(lo), float(hi))
        key = (features.get("zona_omi"), features.get("tipologia"), features.get("stato"))
        return self._band_index.get(key)

    def score(
        self,
        features: dict[str, Any],
        actual_price_per_m2: float | None = None,
    ) -> dict[str, Any]:
        predicted = round(self.predict_price_per_m2(features), 4)
        out: dict[str, Any] = {
            "predicted_price_per_m2_monthly": predicted,
            "features_used": list(FEATURE_COLS),
            "model_path": str(self.model_path),
            "omi_loc_min": None,
            "omi_loc_max": None,
            "omi_half_width": None,
            "band_source": None,
        }
        band = self._resolve_omi_band(features)
        if band is not None:
            out.update(band)
        if actual_price_per_m2 is not None:
            gap_pct = round((actual_price_per_m2 - predicted) / predicted, 4)
            out["price_per_m2_monthly"] = actual_price_per_m2
            out["gap_pct"] = gap_pct
            out["deal_label"] = self.classify_deal(actual_price_per_m2, predicted)
        return out
