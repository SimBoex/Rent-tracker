"""HTTP client for the Render (or local) predict API — used by the Streamlit UI."""

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


def ingest_omi(
    api_base: str,
    *,
    filename: str,
    content: bytes,
    token: str,
    run_pipeline: bool = False,
    timeout_s: float = 120.0,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """POST /ingest/omi with X-Ingest-Token (admin CSV upload)."""
    url = f"{api_base.rstrip('/')}/ingest/omi"
    headers = {"X-Ingest-Token": token}
    files = {"file": (filename, content, "text/csv")}
    data = {"run_pipeline": "true" if run_pipeline else "false"}
    if client is not None:
        resp = client.post(url, headers=headers, files=files, data=data)
        if resp.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"{resp.status_code}: {resp.text}",
                request=resp.request,
                response=resp,
            )
        return resp.json()
    with httpx.Client(timeout=timeout_s) as owned:
        resp = owned.post(url, headers=headers, files=files, data=data)
        if resp.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"{resp.status_code}: {resp.text}",
                request=resp.request,
                response=resp,
            )
        return resp.json()