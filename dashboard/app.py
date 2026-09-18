"""Streamlit dashboard: profile history + predict (SHAP) + monitoring (RF-10)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from dashboard.api_client import health as api_health
from dashboard.api_client import ingest_omi as api_ingest_omi
from dashboard.api_client import create_sighting
from dashboard.api_client import profile_history as api_profile_history
from dashboard.api_client import resolve_api_base_url
from dashboard.api_client import tipologias as api_tipologias
from dashboard.api_client import zones as api_zones
from api.tipologie import list_tipologie as tipologia_options_from_features
from api.zone import list_zones as zone_options_from_features
from api.zone import zone_label

DEFAULT_DRIFT_SUMMARY = _ROOT / "reports" / "drift_latest" / "summary.json"
DEFAULT_RETRAIN_DECISION = _ROOT / "reports" / "retrain_latest" / "decision.json"
DEFAULT_MODEL_PATH = _ROOT / "models" / "baseline_latest" / "model.joblib"
DEFAULT_SIGHTINGS_DRIFT = _ROOT / "reports" / "sightings_drift_latest" / "summary.json"

STATO_OPTIONS = ["OTTIMO", "NORMALE", "SCADENTE"]
STATO_LABELS = {
    "OTTIMO": "Excellent",
    "NORMALE": "Normal",
    "SCADENTE": "Poor",
}
TIPOLOGIA_LABELS = {
    "Abitazioni civili": "Ordinary residential",
    "Abitazioni di tipo economico": "Economy residential",
    "Abitazioni signorili": "Prestigious residential",
    "Box": "Garage (box)",
    "Capannoni industriali": "Industrial warehouses",
    "Capannoni tipici": "Typical warehouses",
    "Laboratori": "Workshops / labs",
    "Negozi": "Shops",
    "Posti auto coperti": "Covered parking",
    "Posti auto scoperti": "Open parking",
    "Uffici": "Offices",
    "Ville e Villini": "Villas",
}
FEATURE_LABELS = {
    "zona_omi": "OMI zone",
    "tipologia": "typology",
    "stato": "condition",
    "loc_mid_lag": "prior-semester mid",
}



def _tipologia_label(value: str) -> str:
    return TIPOLOGIA_LABELS.get(value, value)


def _stato_label(value: str) -> str:
    return STATO_LABELS.get(value, value)


st.set_page_config(page_title="Roma Rent Monitor", layout="wide")
st.title("Roma Rent Monitor")
st.caption(
    "OMI fair-rent benchmark · profile history · predict + SHAP · monitoring · "
    "Source: «Agenzia Entrate – OMI»"
)


@st.cache_data(ttl=3600)
def _tipologia_options(api_base: str | None) -> list[str]:
    # Do not catch API errors here: a swallowed failure would cache [] for ttl.
    if api_base:
        return api_tipologias(api_base)
    return tipologia_options_from_features()


@st.cache_data(ttl=3600)
def _zone_options(api_base: str | None) -> list[dict[str, Any]]:
    # Do not catch API errors here: a swallowed failure would cache [] for ttl.
    if api_base:
        return api_zones(api_base)
    return zone_options_from_features()


def _zona_selectbox(zones: list[dict[str, Any]], *, key: str) -> tuple[str, str]:
    """Select zona_omi; returns (code, display_label)."""
    codes = [str(z["zona_omi"]) for z in zones]
    labels = {
        str(z["zona_omi"]): str(z.get("label") or zone_label(z["zona_omi"], z.get("descr")))
        for z in zones
    }
    default_idx = codes.index("B12") if "B12" in codes else 0
    code = st.selectbox(
        "OMI zone",
        codes,
        index=default_idx,
        format_func=lambda c: labels.get(c, c),
        key=key,
    )
    return code, labels.get(code, code)


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

def _sightings_monitoring_block() ->  None:
    drift = _load_json(DEFAULT_SIGHTINGS_DRIFT)
    st.subheader("Sighting monitoring")
    st.caption("asking is not OMI mid-line — it's a comparison with the model prediction.")
    if drift is None:
        st.info("No sightings drift summary in the latest monitoring snapshot.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("n_samples", drift.get("n_samples"))
        
        c2.metric("mae", f"{drift.get('mae'):.2f}")
        c3.metric("bias", f"{drift.get('bias'):.2f}")

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


def _profile_label(zona_omi: str, tipologia: str, stato: str) -> str:
    return f"{zona_omi} · {_tipologia_label(tipologia)} · {_stato_label(stato)}"


def _profile_history_block() -> None:
    st.subheader("Profile history")
    st.caption(
        "Add one or more OMI zone / typology / condition profiles to compare "
        "semester mid lines and next-semester forecasts. Via API only."
    )
    api_base = resolve_api_base_url()
    if not api_base:
        st.info("Set `RENT_API_URL` to load profile history from the API.")
        return

    try:
        tipologia_opts = _tipologia_options(api_base)
    except Exception as exc:
        st.error(f"Cannot load typologies from API: {exc}")
        return
    if not tipologia_opts:
        st.error(
            "No typologies available (API `/meta/tipologie` empty or "
            "`features_latest.jsonl` missing)."
        )
        return

    try:
        zone_opts = _zone_options(api_base)
    except Exception as exc:
        st.error(f"Cannot load zones from API: {exc}")
        return
    if not zone_opts:
        st.error(
            "No zones available (API `/meta/zones` empty or "
            "`features_latest.jsonl` missing)."
        )
        return

    if "profile_compare" not in st.session_state:
        st.session_state.profile_compare = []

    c1, c2, c3 = st.columns(3)
    with c1:
        zona_omi, zona_label_ui = _zona_selectbox(zone_opts, key="profile_zona")
    with c2:
        tipologia = st.selectbox(
            "typology",
            tipologia_opts,
            index=0,
            format_func=_tipologia_label,
            key="profile_tipologia",
        )
    with c3:
        stato = st.selectbox(
            "condition",
            STATO_OPTIONS,
            index=1,
            format_func=_stato_label,
            key="profile_stato",
        )

    b1, b2 = st.columns(2)
    add_clicked = b1.button("Add profile", type="primary", key="profile_add")
    clear_clicked = b2.button("Clear all", key="profile_clear")

    if clear_clicked:
        st.session_state.profile_compare = []

    if add_clicked:
        try:
            result = api_profile_history(
                api_base,
                zona_omi=zona_omi,
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
        else:
            label = _profile_label(zona_label_ui, tipologia, stato)
            entry = {
                "label": label,
                "series": result.get("series") or [],
                "test_semester": result.get("test_semester"),
                "next_prediction": result.get("next_prediction") or {},
                "source_attribution": result.get(
                    "source_attribution", "Agenzia Entrate – OMI"
                ),
            }
            profiles = [
                p for p in st.session_state.profile_compare if p["label"] != label
            ]
            profiles.append(entry)
            st.session_state.profile_compare = profiles

    profiles: list[dict[str, Any]] = st.session_state.profile_compare
    if not profiles:
        st.info("Add at least one profile to plot.")
        return

    import pandas as pd

    series_cols: dict[str, dict[str, float]] = {}
    summary_rows: list[dict[str, Any]] = []
    for profile in profiles:
        series = profile["series"]
        series_cols[profile["label"]] = {
            str(p["semester"]): float(p["price_per_m2_monthly"])
            for p in series
            if p.get("semester") is not None and p.get("price_per_m2_monthly") is not None
        }
        test_mid = next(
            (p.get("price_per_m2_monthly") for p in series if p.get("role") == "test"),
            None,
        )
        nxt = profile.get("next_prediction") or {}
        pred = nxt.get("predicted_price_per_m2_monthly")
        summary_rows.append(
            {
                "profile": profile["label"],
                "latest semester": profile.get("test_semester") or "—",
                "latest mid €/m²": None if test_mid is None else round(float(test_mid), 2),
                "next fair €/m²": None if pred is None else round(float(pred), 2),
            }
        )

    chart_df = pd.DataFrame(series_cols).sort_index()
    st.line_chart(chart_df)
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    st.caption(profiles[-1].get("source_attribution", "Agenzia Entrate – OMI"))

    remove = st.multiselect(
        "Remove profiles",
        options=[p["label"] for p in profiles],
        key="profile_remove",
    )
    if remove and st.button("Remove selected", key="profile_remove_btn"):
        st.session_state.profile_compare = [
            p for p in profiles if p["label"] not in set(remove)
        ]
        st.rerun()


def _try_predict_block() -> None:
    st.subheader("Listing you saw")
    st.caption(
        "Fair €/m² for an OMI zone / typology / condition,"
        "enter the asking rent ÷ m² from a portal ad (or price/m² you observed)."
    )

    _sightings_monitoring_block()
    api_base = resolve_api_base_url()
    if api_base:
        try:
            h = api_health(api_base)
            if not h.get("model_loaded"):
                st.warning("API reachable but model not loaded (`/health` degraded).")
        except Exception as exc:
            st.error(f"Cannot reach API: {exc}")
            return
    else:
        st.caption(
            "Set `RENT_API_URL` to submit listings to the API."
        )
        return

    try:
        tipologia_opts = _tipologia_options(api_base)
    except Exception as exc:
        st.error(f"Cannot load typologies from API: {exc}")
        return
    if not tipologia_opts:
        st.error(
            "No typologies available (API `/meta/tipologie` empty or "
            "`features_latest.jsonl` missing)."
        )
        return

    try:
        zone_opts = _zone_options(api_base)
    except Exception as exc:
        st.error(f"Cannot load zones from API: {exc}")
        return
    if not zone_opts:
        st.error(
            "No zones available (API `/meta/zones` empty or "
            "`features_latest.jsonl` missing)."
        )
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        zona_omi, _ = _zona_selectbox(zone_opts, key="predict_zona")
        tipologia = st.selectbox(
            "typology",
            tipologia_opts,
            index=0,
            format_func=_tipologia_label,
            key="predict_tipologia",
        )
    with c2:
        stato = st.selectbox(
            "condition",
            STATO_OPTIONS,
            index=1,
            format_func=_stato_label,
            key="predict_stato",
        )
    with c3:
        actual = st.number_input(
            "asking €/m²",
            min_value=0.0,
            value=0.0,
            step=0.5,
            help=(
                "Asking rent ÷ m² from a portal ad (or price/m² you observed). "
                "Deal label uses OMI min–max when available; otherwise ±10% vs model fair."
            ),
            key="predict_asking",
        )


    
    if not st.button("Submit", type="primary", key="submit_sighting"):
        return
    
    if actual <= 0:
        st.error("Enter a valid asking rent ÷ m².")
        return

    payload: dict[str, Any] = {
        "zona_omi": zona_omi,
        "tipologia": tipologia,
        "stato": stato,
        "asking_eur_m2": float(actual),
    }
    try:
        if api_base:
            result = create_sighting(api_base, payload)
        else:
            st.error("No API base — cannot create sighting.")
            return
    except Exception as exc:
        st.error(f"Predict failed: {exc}")
        return
    st.success(f"Sighting submitted: {result['sighting_id']} at {result['submitted_at']}")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("fair €/m² (model)", f"{result['predicted_price_per_m2_monthly']:.2f}")
    m2.metric("vs asking", result.get("deal_label") or "—")
    gap = result.get("gap_pct")
    m3.metric("gap vs fair", "—" if gap is None else f"{100.0 * float(gap):.1f}%")
    lo, hi = result.get("omi_loc_min"), result.get("omi_loc_max")
    half = result.get("omi_half_width")
    basis = result.get("deal_basis")
    if result.get("deal_label") is not None:
        if basis == "omi_band" and lo is not None and hi is not None:
            st.caption(
                f"Deal criterion: OMI locazione band {float(lo):.1f}–{float(hi):.1f} €/m² "
                f"(latest semester, half-width {float(half):.1f}) — «Agenzia Entrate – OMI». "
                "below / in / above = asking vs that official min–max."
            )
        elif basis == "model_pct":
            st.caption(
                "Deal criterion: ±10% vs model fair €/m² "
                "(no OMI locazione band for this zone / typology / condition). "
                "gap vs fair still uses the model prediction."
            )
        elif lo is not None and hi is not None:
            st.caption(
                f"OMI band (latest semester): {float(lo):.1f}–{float(hi):.1f} €/m² "
                f"(half-width {float(half):.1f}) — «Agenzia Entrate – OMI»"
            )
    elif lo is not None and hi is not None:
        st.caption(
            f"OMI band (latest semester): {float(lo):.1f}–{float(hi):.1f} €/m² "
            f"(half-width {float(half):.1f}) — «Agenzia Entrate – OMI»"
        )

    shap_rows = result.get("shap_values") or []
    if shap_rows:

        with st.expander("Why this fair rent? (SHAP)", expanded=False):
            shap_df = pd.DataFrame(shap_rows)
            shap_df["feature"] = shap_df["feature"].map(
                lambda f: FEATURE_LABELS.get(str(f), str(f))
            )
            chart_df = shap_df.set_index("feature")[["shap_value"]]
            st.bar_chart(chart_df)
            base = result.get("shap_base_value")
            if base is not None:
                st.caption(
                    f"SHAP base value E[f(x)] ≈ {float(base):.2f} €/m² · "
                    "bars = contribution toward the prediction above/below that baseline."
                )
    
    

def _admin_ingest_block() -> None:
    with st.expander("Admin", expanded=False):

        _metric_block(*_resolve_monitoring())
        st.divider()   
        
        st.subheader("Ingest OMI semester CSV")
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


_profile_history_block()
st.divider()
_try_predict_block()
st.divider()
_admin_ingest_block()
st.divider()
st.caption(
    "Quotes and zone structure: «Agenzia Entrate – OMI». "
    "This UI does not redistribute raw OMI CSV dumps."
)
