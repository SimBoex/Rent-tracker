"""Gradio try-predict UI for HF Spaces → Render API (option A)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import gradio as gr

from dashboard.api_client import health, predict, resolve_api_base_url

MUNICIPIO_OPTIONS = [
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII",
    "IX", "X", "XI", "XII", "XIII", "XIV", "XV",
]


def _run_predict(
    surface_m2: float,
    rooms: float,
    distance_from_center_km: float,
    area_price_per_m2_hist: float,
    publication_month: int,
    municipio: str,
    price_per_m2_monthly: float,
) -> tuple[str, str, str]:
    api_base = resolve_api_base_url()
    if not api_base:
        return (
            "Set Space secret/variable `RENT_API_URL` to your Render URL "
            "(e.g. https://….onrender.com).",
            "—",
            "—",
        )
    try:
        h = health(api_base)
        if not h.get("model_loaded"):
            return "API reachable but model not loaded (`/health` degraded).", "—", "—"
    except Exception as exc:
        return f"Cannot reach API ({api_base}): {exc}", "—", "—"

    payload = {
        "surface_m2": float(surface_m2),
        "rooms": float(rooms),
        "distance_from_center_km": float(distance_from_center_km),
        "area_price_per_m2_hist": float(area_price_per_m2_hist),
        "publication_month": int(publication_month),
        "municipio": municipio,
    }
    if price_per_m2_monthly and float(price_per_m2_monthly) > 0:
        payload["price_per_m2_monthly"] = float(price_per_m2_monthly)

    try:
        result = predict(api_base, payload)
    except Exception as exc:
        return f"Predict failed: {exc}", "—", "—"

    pred = f"{result['predicted_price_per_m2_monthly']:.2f}"
    label = result.get("deal_label") or "—"
    gap = result.get("gap_pct")
    gap_s = "—" if gap is None else f"{100.0 * float(gap):.1f}%"
    return pred, label, gap_s


demo = gr.Interface(
    fn=_run_predict,
    inputs=[
        gr.Number(label="surface_m2", value=70, minimum=1),
        gr.Number(label="rooms", value=2, minimum=1),
        gr.Number(label="distance_from_center_km", value=3.0, minimum=0),
        gr.Number(label="area_price_per_m2_hist", value=25.0, minimum=0),
        gr.Slider(label="publication_month", minimum=1, maximum=12, step=1, value=9),
        gr.Dropdown(label="municipio", choices=MUNICIPIO_OPTIONS, value="I"),
        gr.Number(
            label="price_per_m2_monthly (optional, 0 = skip deal label)",
            value=0,
            minimum=0,
        ),
    ],
    outputs=[
        gr.Textbox(label="predicted €/m²"),
        gr.Textbox(label="deal_label"),
        gr.Textbox(label="gap_pct"),
    ],
    title="Roma Rent Monitor",
    description=(
        "Fair rent €/m² via the Render FastAPI backend. "
        "Set `RENT_API_URL` in Space secrets."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
