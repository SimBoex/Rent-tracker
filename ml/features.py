"""Shared OMI feature / target column names (train, pipeline, serve, drift)."""

from __future__ import annotations

TARGET = "price_per_m2_monthly"
# OMI Open Data features — see doc/omi.md / SOR2 RF-01d
NUMERIC_FEATURES = ["loc_mid_lag"]
CATEGORICAL_FEATURES = ["zona_omi", "tipologia", "stato"]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
