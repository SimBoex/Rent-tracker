"""Streamlit dashboard: try-predict via API + monitoring + good deals (RF-10)."""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit runs this file without installing the repo; keep imports like `api` / `ml` working.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from api.predictor import DEFAULT_MODEL_PATH, ModelPredictor
from dashboard.api_client import health as api_health
from dashboard.api_client import predict as api_predict
from dashboard.api_client import resolve_api_base_url
from dashboard.data import (
    DEFAULT_DRIFT_SUMMARY,
    DEFAULT_RETRAIN_DECISION,
    good_deals_table,
    load_feature_rows,
    load_json,
    score_listings,
)
from ml.train import DEFAULT_INPUT, FEATURE_COLS

MUNICIPIO_OPTIONS = [
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII",
    "IX", "X", "XI", "XII", "XIII", "XIV", "XV",
]

st.set_page_config(page_title="Roma Rent Monitor", layout="wide")
st.title("Roma Rent Monitor")
st.caption("Try a listing via the predict API · monitoring · good deals.")


def _try_predict_block() -> None:
    st.subheader("Try a prediction")
    api_base = resolve_api_base_url()
    if api_base:
        st.caption(f"Backend: `{api_base}`")
        try:
            h = api_health(api_base)
            if not h.get("model_loaded"):
                st.warning("API reachable but model not loaded (`/health` degraded).")
        except Exception as exc:
            st.error(f"Cannot reach API: {exc}")
            return
    else:
        st.caption("No `RENT_API_URL` — using local `models/baseline_latest` if present.")

    c1, c2, c3 = st.columns(3)
    with c1:
        surface_m2 = st.number_input("surface_m2", min_value=1.0, value=70.0, step=1.0)
        rooms = st.number_input("rooms", min_value=1.0, value=2.0, step=1.0)
    with c2:
        distance = st.number_input("distance_from_center_km", min_value=0.0, value=3.0, step=0.1)
        area_hist = st.number_input("area_price_per_m2_hist", min_value=0.0, value=25.0, step=0.5)
    with c3:
        month = st.number_input("publication_month", min_value=1, max_value=12, value=9, step=1)
        municipio = st.selectbox("municipio", MUNICIPIO_OPTIONS, index=0)
        actual = st.number_input(
            "price_per_m2_monthly (optional, for deal label)",
            min_value=0.0,
            value=0.0,
            step=0.5,
            help="Leave 0 to skip RF-07 deal classification.",
        )

    if not st.button("Predict", type="primary"):
        return

    payload = {
        "surface_m2": float(surface_m2),
        "rooms": float(rooms),
        "distance_from_center_km": float(distance),
        "area_price_per_m2_hist": float(area_hist),
        "publication_month": int(month),
        "municipio": municipio,
    }
    if actual and actual > 0:
        payload["price_per_m2_monthly"] = float(actual)

    try:
        if api_base:
            result = api_predict(api_base, payload)
        else:
            model_path = Path(DEFAULT_MODEL_PATH)
            if not model_path.is_file():
                st.error(f"Model not found at `{model_path}`. Set `RENT_API_URL` or train locally.")
                return
            features = {c: payload.get(c) for c in FEATURE_COLS}
            result = ModelPredictor(model_path).score(
                features,
                actual_price_per_m2=payload.get("price_per_m2_monthly"),
            )
    except Exception as exc:
        st.error(f"Predict failed: {exc}")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("predicted €/m²", f"{result['predicted_price_per_m2_monthly']:.2f}")
    m2.metric("deal_label", result.get("deal_label") or "—")
    gap = result.get("gap_pct")
    m3.metric("gap_pct", "—" if gap is None else f"{100.0 * float(gap):.1f}%")


def _metric_block(drift: dict | None, decision: dict | None) -> None:
    st.subheader("Monitoring")
    if drift is None:
        st.info(
            f"No drift summary at `{DEFAULT_DRIFT_SUMMARY}` "
            "(expected on HF Spaces — run monitoring locally / CI)."
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("n_reference", drift.get("n_reference"))
        c2.metric("n_current", drift.get("n_current"))
        share = drift.get("drifted_columns_share")
        c3.metric("drifted_share", None if share is None else f"{float(share):.2%}")
        mae_ref = (drift.get("mae_reference") or {}).get("mae")
        mae_cur = (drift.get("mae_current") or {}).get("mae")
        c4.metric(
            "MAE ref → cur",
            "—" if mae_ref is None or mae_cur is None else f"{mae_ref:.2f} → {mae_cur:.2f}",
        )
        mae_by_day = drift.get("mae_by_day")
        if mae_by_day:
            st.write("MAE by scrape day")
            st.dataframe(mae_by_day, use_container_width=True, hide_index=True)

    if decision is None:
        st.info(f"No retrain decision at `{DEFAULT_RETRAIN_DECISION}`.")
    else:
        st.write(
            f"**Retrain gate:** `should_retrain={decision.get('should_retrain')}` · "
            f"reason=`{decision.get('trigger_reason')}` · "
            f"mae_ratio=`{decision.get('mae_ratio')}`"
        )


def _deals_block() -> None:
    st.subheader("Good deals")
    model_path = Path(DEFAULT_MODEL_PATH)
    features_path = Path(DEFAULT_INPUT)
    if not model_path.is_file() or not features_path.is_file():
        st.info(
            "Good-deal table needs local `features_latest` + `baseline_latest` "
            "(available after pipeline locally; not shipped to HF Spaces)."
        )
        return

    rows = load_feature_rows(features_path)
    if not rows:
        st.warning("No scorable rows in features file.")
        return

    predictor = ModelPredictor(model_path)
    scored = score_listings(rows, predictor)
    deals = good_deals_table(scored)
    st.write(
        f"Scored **{len(scored)}** listings · **{len(deals)}** good deals "
        f"(actual €/m² ≤ predicted − 10%)."
    )
    if deals.empty:
        st.info("No good deals in the current sample.")
    else:
        st.dataframe(deals, use_container_width=True, hide_index=True)


_try_predict_block()
st.divider()
_metric_block(load_json(DEFAULT_DRIFT_SUMMARY), load_json(DEFAULT_RETRAIN_DECISION))
st.divider()
_deals_block()
