"""Streamlit dashboard: monitoring metrics + good-deal listings (RF-10 lite)."""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit runs this file without installing the repo; keep imports like `api` / `ml` working.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from api.predictor import DEFAULT_MODEL_PATH, ModelPredictor
from dashboard.data import (
    DEFAULT_DRIFT_SUMMARY,
    DEFAULT_RETRAIN_DECISION,
    good_deals_table,
    load_feature_rows,
    load_json,
    score_listings,
)
from ml.train import DEFAULT_INPUT

st.set_page_config(page_title="Roma Rent Monitor", layout="wide")
st.title("Roma Rent Monitor")
st.caption("RF-10 lite — drift / retrain metrics and good-deal listings (local).")


def _metric_block(drift: dict | None, decision: dict | None) -> None:
    st.subheader("Monitoring")
    if drift is None:
        st.warning(f"No drift summary at `{DEFAULT_DRIFT_SUMMARY}`. Run `python -m ml.drift_report`.")
    else:
        # st.columns returns a tuple of columns
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
        st.info(f"No retrain decision at `{DEFAULT_RETRAIN_DECISION}`. Run `python -m ml.retrain_check --dry-run`.")
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
    if not model_path.is_file():
        st.warning(f"Model not found at `{model_path}`. Train with `python -m ml.train`.")
        return
    if not features_path.is_file():
        st.warning(f"Features not found at `{features_path}`. Run the pipeline / features phase.")
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


_metric_block(load_json(DEFAULT_DRIFT_SUMMARY), load_json(DEFAULT_RETRAIN_DECISION))
st.divider()
_deals_block()
