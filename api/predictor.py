"""Load baseline model and score OMI zone rows (RF-06 / RF-07)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from ml.train import FEATURE_COLS, MODELS_DIR

DEFAULT_MODEL_PATH = MODELS_DIR / "baseline_latest" / "model.joblib"

# Relative gap vs predicted fair €/m²: |actual - pred| / pred
DEAL_BAND = 0.10

BELOW_OMI_BAND = "below_omi_band"
IN_BAND = "in_band"
ABOVE_OMI_BAND = "above_omi_band"
# Back-compat alias used by dashboard filters
GOOD_DEAL = BELOW_OMI_BAND


class ModelPredictor:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
        loaded = joblib.load(model_path)
        if not isinstance(loaded, Pipeline):
            raise TypeError(f"Expected sklearn Pipeline, got {type(loaded)!r}")
        self.pipeline: Pipeline = loaded
        self.model_path = model_path

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
        }
        if actual_price_per_m2 is not None:
            gap_pct = round((actual_price_per_m2 - predicted) / predicted, 4)
            out["price_per_m2_monthly"] = actual_price_per_m2
            out["gap_pct"] = gap_pct
            out["deal_label"] = self.classify_deal(actual_price_per_m2, predicted)
        return out
