"""Temporal train/test (and drift reference/current) split by OMI semester."""

from __future__ import annotations

from typing import Any

from etl.semester import semester_key

# Re-export for callers that import semester_key from ml.split
__all__ = ["listing_sort_key", "semester_key", "temporal_split"]


def listing_sort_key(row: dict[str, Any]) -> tuple[int, int]:
    semester = row.get("semester")
    if semester is None:
        raise ValueError(f"Missing semester in row: {row}")
    return semester_key(str(semester))


def temporal_split(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Last OMI semester = test / current. Requires ≥2 distinct semesters."""
    if not rows:
        return [], [], "empty"

    ordered = sorted(rows, key=listing_sort_key)
    semesters = {row.get("semester") for row in ordered}
    last = ordered[-1].get("semester")
    if len(semesters) >= 2:
        train = [r for r in ordered if r.get("semester") != last]
        test = [r for r in ordered if r.get("semester") == last]
        if train and test:
            return train, test, "temporal_last_semester"

    raise ValueError(f"Not enough semesters to split: {semesters}")
