"""Load monitoring JSON and score listings for the Streamlit dashboard (RF-10)."""

from __future__ import annotations

from typing import Any
from pathlib import Path
import json

import pandas as pd

from api.predictor import GOOD_DEAL, ModelPredictor
from ml.drift_report import REPORTS_DIR
from ml.train import DEFAULT_INPUT, FEATURE_COLS, TARGET

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRIFT_SUMMARY = REPORTS_DIR / "drift_latest" / "summary.json"
DEFAULT_RETRAIN_DECISION = REPORTS_DIR / "retrain_latest" / "decision.json"
MAX_SCORE_ROWS = 2000

DISPLAY_COLS = [
    "listing_id",
    "url",
    "municipio",
    "surface_m2",
    "rooms",
    TARGET,
    "predicted_price_per_m2_monthly",
    "gap_pct",
    "deal_label",
]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_feature_rows(path: Path = DEFAULT_INPUT, limit: int = MAX_SCORE_ROWS) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get(TARGET) is None:
            continue
        municipio = row.get("municipio")
        if municipio is None or municipio == "":
            continue
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def score_listings(
    rows: list[dict[str, Any]],
    predictor: ModelPredictor,
) -> pd.DataFrame:
    scored: list[dict[str, Any]] = []
    for row in rows:
        features = {c: row.get(c) for c in FEATURE_COLS}
        # predictor.score takes features and actual_price_per_m2 and returns a dictionary with the predicted price per m2 monthly, gap percentage, and deal label
        result = predictor.score(features, actual_price_per_m2=float(row[TARGET]))
        scored.append(
            {
                "listing_id": row.get("listing_id"),
                "url": row.get("url"),
                "municipio": row.get("municipio"),
                "surface_m2": row.get("surface_m2"),
                "rooms": row.get("rooms"),
                TARGET: float(row[TARGET]),
                "predicted_price_per_m2_monthly": result["predicted_price_per_m2_monthly"],
                "gap_pct": result["gap_pct"],
                "deal_label": result["deal_label"],
            }
        )
    if not scored:
        return pd.DataFrame(columns=DISPLAY_COLS)
    return pd.DataFrame(scored)


def good_deals_table(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty or "deal_label" not in scored.columns:
        return pd.DataFrame(columns=DISPLAY_COLS)
    deals = scored[scored["deal_label"] == GOOD_DEAL].copy()
    return deals.sort_values("gap_pct", ascending=True).reset_index(drop=True)
