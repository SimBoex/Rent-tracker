"""Streamlit dashboard: try-predict via API + monitoring + zone deals (RF-10)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from dashboard.api_client import health as api_health
from dashboard.api_client import predict as api_predict
from dashboard.api_client import resolve_api_base_url

DEFAULT_DRIFT_SUMMARY = _ROOT / "reports" / "drift_latest" / "summary.json"
DEFAULT_RETRAIN_DECISION = _ROOT / "reports" / "retrain_latest" / "decision.json"
DEFAULT_MODEL_PATH = _ROOT / "models" / "baseline_latest" / "model.joblib"
DEFAULT_INPUT = _ROOT / "data" / "processed" / "features_latest.jsonl"

TIPOLOGIA_OPTIONS = [
    "Abitazioni civili",
    "Abitazioni signorili",
    "Abitazioni di tipo economico",
]
STATO_OPTIONS = ["OTTIMO", "NORMALE", "SCADENTE"]

st.set_page_config(page_title="Roma Rent Monitor", layout="wide")
st.title("Roma Rent Monitor")
st.caption(
    "OMI zone fair-rent demo · monitoring · below-band flags · "
    "Source: «Agenzia Entrate – OMI»"
)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
        if "<<<<<<<" in text:
            return None
        return json.loads(text)
    except (json.JSONDecodeError, OSError):
        return None


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
        zona_omi = st.text_input("zona_omi", value="B12")
        tipologia = st.selectbox("tipologia", TIPOLOGIA_OPTIONS, index=0)
    with c2:
        stato = st.selectbox("stato", STATO_OPTIONS, index=1)
        loc_mid_lag = st.number_input(
            "loc_mid_lag (prior semester mid, optional)",
            min_value=0.0,
            value=0.0,
            step=0.5,
            help="Leave 0 if no previous semester.",
        )
    with c3:
        month = st.number_input("publication_month", min_value=1, max_value=12, value=6, step=1)
        actual = st.number_input(
            "price_per_m2_monthly (optional, for deal label)",
            min_value=0.0,
            value=0.0,
            step=0.5,
            help="Observed OMI mid €/m² to classify vs model.",
        )

    if not st.button("Predict", type="primary"):
        return

    payload: dict[str, Any] = {
        "zona_omi": zona_omi.strip(),
        "tipologia": tipologia,
        "stato": stato,
        "publication_month": int(month),
    }
    if loc_mid_lag and loc_mid_lag > 0:
        payload["loc_mid_lag"] = float(loc_mid_lag)
    if actual and actual > 0:
        payload["price_per_m2_monthly"] = float(actual)

    try:
        if api_base:
            result = api_predict(api_base, payload)
        else:
            from api.predictor import ModelPredictor
            from ml.train import FEATURE_COLS

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
    if drift is None and decision is None:
        st.info(
            "Monitoring snapshot not available yet. "
            "After the next OMI CI run, drift / retrain metrics appear here."
        )
        return
    if drift is None:
        st.info("No drift summary in the latest monitoring snapshot.")
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
            st.write("MAE by semester day")
            st.dataframe(mae_by_day, use_container_width=True, hide_index=True)

    if decision is None:
        st.info("No retrain decision in the latest monitoring snapshot.")
    else:
        st.write(
            f"**Retrain gate:** `should_retrain={decision.get('should_retrain')}` · "
            f"reason=`{decision.get('trigger_reason')}` · "
            f"mae_ratio=`{decision.get('mae_ratio')}`"
        )


def _resolve_monitoring() -> tuple[dict | None, dict | None]:
    drift = _load_json(DEFAULT_DRIFT_SUMMARY)
    decision = _load_json(DEFAULT_RETRAIN_DECISION)
    if drift is not None or decision is not None:
        return drift, decision
    from dashboard.snapshots import load_monitoring_snapshot

    snap = load_monitoring_snapshot()
    if not snap:
        return None, None
    return snap.get("drift"), snap.get("decision")


def _deals_block() -> None:
    st.subheader("Below-band zones")
    st.caption(
        "Public view: zone / typology / state / semester + deal label only — "
        "no OMI €/m² values. Source: «Agenzia Entrate – OMI»."
    )
    model_path = Path(DEFAULT_MODEL_PATH)
    features_path = Path(DEFAULT_INPUT)

    if model_path.is_file() and features_path.is_file():
        from api.predictor import ModelPredictor
        from dashboard.data import (
            PUBLIC_DISPLAY_COLS,
            good_deals_table,
            load_feature_rows,
            score_rows,
        )

        rows = load_feature_rows(features_path)
        if not rows:
            st.warning("No scorable rows in features file.")
            return
        predictor = ModelPredictor(model_path)
        scored = score_rows(rows, predictor)
        deals = good_deals_table(scored)
        public = deals.reindex(columns=PUBLIC_DISPLAY_COLS)
        st.write(
            f"Scored **{len(scored)}** rows · **{len(public)}** below model band "
            f"(labels only in this table)."
        )
        if public.empty:
            st.info("No below-band rows in the current sample.")
        else:
            st.dataframe(public, use_container_width=True, hide_index=True)
        return

    from dashboard.snapshots import load_good_deals_snapshot

    snap = load_good_deals_snapshot()
    if snap is None:
        st.info(
            "Zone deals snapshot not available yet. "
            "After the next OMI CI run it appears here automatically."
        )
        return
    rows = snap.get("rows") or []
    st.caption(
        f"{snap.get('source_attribution', 'Agenzia Entrate – OMI')} · "
        f"snapshot `{snap.get('generated_at', '?')}` · "
        f"scored={snap.get('n_scored')} · below_band={snap.get('n_good_deals')}"
    )
    if not rows:
        st.info("No below-band rows in the latest snapshot.")
    else:
        st.dataframe(rows, use_container_width=True, hide_index=True)


_try_predict_block()
st.divider()
_metric_block(*_resolve_monitoring())
st.divider()
_deals_block()
st.divider()
st.caption(
    "Quotazioni and zone structure: «Agenzia Entrate – OMI». "
    "This UI does not redistribute raw OMI CSV dumps or locazione €/m² tables."
)
