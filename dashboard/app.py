"""Streamlit dashboard: try-predict via API + monitoring + profile history (RF-10)."""

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
from dashboard.api_client import ingest_omi as api_ingest_omi
from dashboard.api_client import predict as api_predict
from dashboard.api_client import profile_history as api_profile_history
from dashboard.api_client import resolve_api_base_url
from dashboard.api_client import tipologias as api_tipologias
from dashboard.data import tipologia_options_from_features

DEFAULT_DRIFT_SUMMARY = _ROOT / "reports" / "drift_latest" / "summary.json"
DEFAULT_RETRAIN_DECISION = _ROOT / "reports" / "retrain_latest" / "decision.json"
DEFAULT_MODEL_PATH = _ROOT / "models" / "baseline_latest" / "model.joblib"

STATO_OPTIONS = ["OTTIMO", "NORMALE", "SCADENTE"]

st.set_page_config(page_title="Roma Rent Monitor", layout="wide")
st.title("Roma Rent Monitor")
st.caption(
    "OMI fair-rent benchmark · compare a listing €/m² you saw · profile history · "
    "Source: «Agenzia Entrate – OMI»"
)


@st.cache_data(ttl=3600)
def _tipologia_options(api_base: str | None) -> list[str]:
    if api_base:
        try:
            return api_tipologias(api_base)
        except Exception:
            pass
    return tipologia_options_from_features()


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

    tipologia_opts = _tipologia_options(api_base)
    if not tipologia_opts:
        st.error(
            "No tipologias available (API `/meta/tipologie` empty or "
            "`features_latest.jsonl` missing)."
        )
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        zona_omi = st.text_input("zona_omi", value="B12")
        tipologia = st.selectbox("tipologia", tipologia_opts, index=0)
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
        actual = st.number_input(
            "asking €/m² (optional — listing you saw)",
            min_value=0.0,
            value=0.0,
            step=0.5,
            help=(
                "Asking rent ÷ m² from a portal ad (or price/m² you observed). "
                "Compared to model fair for this zona/tipologia/stato — not an OMI mid lookup."
            ),
        )

    if not st.button("Predict", type="primary"):
        return

    payload: dict[str, Any] = {
        "zona_omi": zona_omi.strip(),
        "tipologia": tipologia,
        "stato": stato,
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
    m1.metric("fair €/m² (model)", f"{result['predicted_price_per_m2_monthly']:.2f}")
    m2.metric("vs asking", result.get("deal_label") or "—")
    gap = result.get("gap_pct")
    m3.metric("gap vs fair", "—" if gap is None else f"{100.0 * float(gap):.1f}%")
    lo, hi = result.get("omi_loc_min"), result.get("omi_loc_max")
    half = result.get("omi_half_width")
    if lo is not None and hi is not None:
        st.caption(
            f"OMI band (latest semester): {float(lo):.1f}–{float(hi):.1f} €/m² "
            f"(half-width {float(half):.1f}) — «Agenzia Entrate – OMI»"
        )


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
            st.write("MAE by OMI semester")
            st.dataframe(mae_by_day, use_container_width=True, hide_index=True)

    if decision is None:
        st.info("No retrain decision in the latest monitoring snapshot.")
    else:
        status = decision.get("retrain_status") or "unknown"
        st.write(
            f"**Retrain gate:** `should_retrain={decision.get('should_retrain')}` · "
            f"status=`{status}` · "
            f"reason=`{decision.get('trigger_reason')}` · "
            f"mae_ratio=`{decision.get('mae_ratio')}`"
        )
        if status == "failed":
            st.error(
                "Retraining failed — baseline unchanged. "
                f"{decision.get('error') or 'see CI logs / decision.json'}"
            )
        elif status == "ok":
            st.success(
                f"Retrain ok → `{decision.get('model_path') or 'baseline_latest'}`"
            )
        elif status == "dry_run":
            st.info("Gate would retrain (dry-run) — train not executed.")


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


def _profile_history_block() -> None:
    st.subheader("Profile history")
    st.caption(
        "Pick zona OMI / tipologia / stato → semester mid history (last = test set) "
        "and model forecast for the next semester. Via API only."
    )
    api_base = resolve_api_base_url()
    if not api_base:
        st.info("Set `RENT_API_URL` to load profile history from the API.")
        return

    tipologia_opts = _tipologia_options(api_base)
    if not tipologia_opts:
        st.error(
            "No tipologias available (API `/meta/tipologie` empty or "
            "`features_latest.jsonl` missing)."
        )
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        zona_omi = st.text_input("zona_omi", value="B12", key="profile_zona")
    with c2:
        tipologia = st.selectbox(
            "tipologia", tipologia_opts, index=0, key="profile_tipologia"
        )
    with c3:
        stato = st.selectbox("stato", STATO_OPTIONS, index=1, key="profile_stato")

    if not st.button("Load history", type="primary", key="profile_load"):
        return

    if not zona_omi.strip():
        st.error("Enter a zona_omi.")
        return

    try:
        result = api_profile_history(
            api_base,
            zona_omi=zona_omi.strip(),
            tipologia=tipologia,
            stato=stato,
        )
    except Exception as exc:
        detail = ""
        resp = getattr(exc, "response", None)
        if resp is not None:
            try:
                detail = resp.json().get("detail") or resp.text
            except Exception:
                detail = getattr(resp, "text", "") or ""
        st.error(
            f"Profile history failed: {exc}"
            + (f" — {detail}" if detail else "")
        )
        return

    series = result.get("series") or []
    nxt = result.get("next_prediction") or {}
    test_sem = result.get("test_semester")

    m1, m2, m3 = st.columns(3)
    m1.metric("test semester (OMI mid)", test_sem or "—")
    test_mid = next(
        (p.get("price_per_m2_monthly") for p in series if p.get("role") == "test"),
        None,
    )
    m2.metric(
        "test mid €/m²",
        "—" if test_mid is None else f"{float(test_mid):.2f}",
    )
    pred = nxt.get("predicted_price_per_m2_monthly")
    m3.metric(
        "next semester fair €/m²",
        "—" if pred is None else f"{float(pred):.2f}",
    )
    lag = nxt.get("loc_mid_lag")
    if lag is not None:
        st.caption(
            f"Next forecast uses `loc_mid_lag` = last mid ({float(lag):.2f}). "
            f"{result.get('source_attribution', 'Agenzia Entrate – OMI')}"
        )

    if series:
        import pandas as pd

        df = pd.DataFrame(series)
        st.line_chart(df.set_index("semester")["price_per_m2_monthly"])
        st.dataframe(
            df[["semester", "price_per_m2_monthly", "role"]],
            use_container_width=True,
            hide_index=True,
        )


def _admin_ingest_block() -> None:
    with st.expander("Admin · upload OMI semester CSV", expanded=False):
        st.caption(
            "Requires API `INGEST_TOKEN`. In cloud the API stores the CSV on R2 "
            "(SHA-256 dedupe) and can trigger GitHub `omi-monitoring`. "
            "Raw CSV is never shown publicly."
        )
        api_base = resolve_api_base_url()
        if not api_base:
            st.warning("Set `RENT_API_URL` to use ingest via the API.")
            return
        token = st.text_input(
            "Ingest token",
            type="password",
            help="Must match API env INGEST_TOKEN",
            key="ingest_token",
        )
        uploaded = st.file_uploader("OMI *VALORI*.csv", type=["csv"], key="omi_upload")
        run_pipe = st.checkbox(
            "After upload: run monitoring (cloud = GitHub Actions; local = pipeline --skip-train)",
            value=True,
        )
        if not st.button("Upload semester", type="secondary"):
            return
        if not token.strip():
            st.error("Enter the ingest token.")
            return
        if uploaded is None:
            st.error("Choose a CSV file.")
            return
        try:
            result = api_ingest_omi(
                api_base,
                filename=uploaded.name,
                content=uploaded.getvalue(),
                token=token.strip(),
                run_pipeline=run_pipe,
            )
        except Exception as exc:
            st.error(f"Ingest failed: {exc}")
            return
        if result.get("status") == "duplicate":
            st.warning(
                f"Duplicate skipped — same content as `{result.get('duplicate_of')}` "
                f"(sha256 `{str(result.get('sha256', ''))[:12]}…`)."
            )
            return
        st.success(
            f"Stored `{result.get('saved_as')}` · semester={result.get('semester')} · "
            f"rows={result.get('n_rows')} · sha256 `{str(result.get('sha256', ''))[:12]}…`"
        )
        if result.get("cloud_key"):
            st.caption(f"R2: `{result['cloud_key']}`")
        if result.get("workflow"):
            st.caption(result["workflow"])
        if result.get("pipeline"):
            st.caption(result["pipeline"])


_try_predict_block()
st.divider()
_metric_block(*_resolve_monitoring())
st.divider()
_profile_history_block()
st.divider()
_admin_ingest_block()
st.divider()
st.caption(
    "Quotazioni and zone structure: «Agenzia Entrate – OMI». "
    "This UI does not redistribute raw OMI CSV dumps."
)
