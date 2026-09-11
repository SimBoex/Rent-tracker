"""Shared synthetic OMI feature rows for unit tests."""

from __future__ import annotations

from typing import Any


def omi_feature_row(
    i: int,
    *,
    day: str = "2026-09-08",
    price: float | None = None,
    loc_mid_lag: float | None = None,
) -> dict[str, Any]:
    month = 6 if int(day[5:7]) <= 6 else 12
    lag = 18.0 + float(i % 7) if loc_mid_lag is None else loc_mid_lag
    return {
        "listing_id": f"row-{i}|{day}",
        "scraped_at": f"{day}T10:00:00+00:00",
        "semester": f"{day[:4]}-{'1' if month == 6 else '2'}",
        "price_per_m2_monthly": 20.0 + i * 0.5 if price is None else price,
        "loc_mid_lag": lag,
        "publication_month": month,
        "zona_omi": "B12" if i % 2 == 0 else "C14",
        "tipologia": "Abitazioni civili",
        "stato": "NORMALE" if i % 2 == 0 else "OTTIMO",
    }
