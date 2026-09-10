"""HTTP client for the Render (or local) predict API — used by HF Spaces (option A)."""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_TIMEOUT_S = 30.0


def resolve_api_base_url(explicit: str | None = None) -> str | None:
    """Prefer explicit arg, then env ``RENT_API_URL``, then Streamlit secrets if available."""
    if explicit and explicit.strip():
        return explicit.strip().rstrip("/")
    env = os.environ.get("RENT_API_URL", "").strip()
    if env:
        return env.rstrip("/")
    try:
        import streamlit as st

        secret = st.secrets.get("RENT_API_URL", "")
        if isinstance(secret, str) and secret.strip():
            return secret.strip().rstrip("/")
    except Exception:
        pass
    return None


def health(
    api_base: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    url = f"{api_base.rstrip('/')}/health"
    if client is not None:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.json()
    with httpx.Client(timeout=timeout_s) as owned:
        resp = owned.get(url)
        resp.raise_for_status()
        return resp.json()


def predict(
    api_base: str,
    payload: dict[str, Any],
    timeout_s: float = DEFAULT_TIMEOUT_S,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    url = f"{api_base.rstrip('/')}/predict"
    if client is not None:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()
    with httpx.Client(timeout=timeout_s) as owned:
        resp = owned.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()
