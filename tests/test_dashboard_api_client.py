"""Unit tests for dashboard API client (Streamlit UI → Render)."""

from __future__ import annotations

import json

import httpx
import pytest

from dashboard.api_client import health, predict, resolve_api_base_url


def test_resolve_api_base_url_explicit_and_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("RENT_API_URL", raising=False)
    assert resolve_api_base_url(" https://api.example.com/ ") == "https://api.example.com"
    monkeypatch.setenv("RENT_API_URL", "https://from-env.onrender.com/")
    assert resolve_api_base_url() == "https://from-env.onrender.com"
    assert resolve_api_base_url("https://override/") == "https://override"


def test_health_and_predict_ok():
    health_body = {"status": "ok", "model_loaded": True, "model_path": "x"}
    pred_body = {
        "predicted_price_per_m2_monthly": 22.5,
        "features_used": ["zona_omi"],
        "model_path": "x",
        "deal_label": "in_band",
        "gap_pct": 0.01,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, json=health_body)
        if request.url.path.endswith("/predict"):
            body = json.loads(request.content.decode())
            assert body["zona_omi"] == "B12"
            return httpx.Response(200, json=pred_body)
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert health("https://api.example.com", client=client) == health_body
        out = predict(
            "https://api.example.com",
            {
                "zona_omi": "B12",
                "tipologia": "Abitazioni civili",
                "stato": "NORMALE",
                "publication_month": 6,
            },
            client=client,
        )
        assert out["predicted_price_per_m2_monthly"] == 22.5


def test_predict_raises_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "Model not loaded"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            predict(
                "https://api.example.com",
                {
                    "zona_omi": "B12",
                    "tipologia": "Abitazioni civili",
                    "stato": "NORMALE",
                    "publication_month": 6,
                },
                client=client,
            )
