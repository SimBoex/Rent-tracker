"""Dispatch GitHub Actions workflow after cloud OMI ingest."""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

DEFAULT_WORKFLOW = "daily_monitoring.yml"


def dispatch_configured() -> bool:
    return bool(
        os.environ.get("GITHUB_TOKEN", "").strip()
        and os.environ.get("GITHUB_REPOSITORY", "").strip()
    )


def dispatch_omi_monitoring(
    *,
    ref: str | None = None,
    workflow: str | None = None,
) -> str:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    repo = os.environ.get("GITHUB_REPOSITORY", "").strip()
    if not token or not repo:
        raise RuntimeError("GITHUB_TOKEN and GITHUB_REPOSITORY required to dispatch")

    wf = (workflow or os.environ.get("INGEST_DISPATCH_WORKFLOW") or DEFAULT_WORKFLOW).strip()
    branch = (ref or os.environ.get("INGEST_DISPATCH_REF") or "main").strip()
    url = f"https://api.github.com/repos/{repo}/actions/workflows/{wf}/dispatches"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, headers=headers, json={"ref": branch})
        if resp.status_code not in {204, 200}:
            raise RuntimeError(
                f"workflow_dispatch failed ({resp.status_code}): {resp.text[:300]}"
            )
    msg = f"dispatched {wf}@{branch} on {repo}"
    logger.info(msg)
    return msg
