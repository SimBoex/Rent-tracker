"""Smoke tests for Gradio try-predict helper."""

from __future__ import annotations

from unittest.mock import patch

from dashboard.gradio_app import _run_predict


def test_run_predict_missing_api(monkeypatch):
    monkeypatch.delenv("RENT_API_URL", raising=False)
    with patch("dashboard.gradio_app.resolve_api_base_url", return_value=None):
        pred, label, gap = _run_predict(70, 2, 3.0, 25.0, 9, "I", 0)
    assert "RENT_API_URL" in pred
    assert label == "—"


def test_run_predict_ok(monkeypatch):
    monkeypatch.setenv("RENT_API_URL", "https://api.example.com")
    with (
        patch("dashboard.gradio_app.resolve_api_base_url", return_value="https://api.example.com"),
        patch("dashboard.gradio_app.health", return_value={"model_loaded": True}),
        patch(
            "dashboard.gradio_app.predict",
            return_value={
                "predicted_price_per_m2_monthly": 22.5,
                "deal_label": "fair_price",
                "gap_pct": 0.05,
            },
        ),
    ):
        pred, label, gap = _run_predict(70, 2, 3.0, 25.0, 9, "I", 20.0)
    assert pred == "22.50"
    assert label == "fair_price"
    assert gap == "5.0%"
