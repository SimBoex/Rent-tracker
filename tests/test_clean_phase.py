"""Unit tests for ListingTransformer clean/dedupe helpers (RNF-05)."""

from __future__ import annotations

import pytest

from etl.transform.clean_phase import ListingTransformer


# ---------------------------------------------------------------------------
# check_title
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "title, expected",
    [
        ("Trilocale via Roma, Roma", False),
        ("Palazzo - Edificio ottimo stato, Roma", True),
        ("Edificio residenziale", True),
        ("Intera proprietà zona Prati", True),
        (None, False),
        ("", False),
    ],
)
def test_check_title(title, expected):
    tr = ListingTransformer()
    assert tr.check_title({"title": title}) is expected


# ---------------------------------------------------------------------------
# dedupe_latest
# ---------------------------------------------------------------------------


def test_dedupe_latest_keeps_newest_scraped_at():
    tr = ListingTransformer()
    rows = [
        {
            "source": "immobiliare.it",
            "listing_id": 1,
            "scraped_at": "2026-09-08T10:00:00+00:00",
            "price_eur_month": 1000,
        },
        {
            "source": "immobiliare.it",
            "listing_id": 1,
            "scraped_at": "2026-09-08T14:00:00+00:00",
            "price_eur_month": 1100,
        },
        {
            "source": "immobiliare.it",
            "listing_id": 2,
            "scraped_at": "2026-09-08T14:00:00+00:00",
            "price_eur_month": 900,
        },
    ]
    out = tr.dedupe_latest(rows)
    assert len(out) == 2
    by_id = {r["listing_id"]: r for r in out}
    assert by_id[1]["price_eur_month"] == 1100


# ---------------------------------------------------------------------------
# clean_row drop reasons / happy path
# ---------------------------------------------------------------------------


def _base_row(**overrides):
    row = {
        "source": "immobiliare.it",
        "listing_id": 42,
        "url": "https://www.immobiliare.it/annunci/42/",
        "title": "Bilocale via Test, Roma",
        "price_eur_month": 1200,
        "surface_m2": "60 m²",
        "rooms": "2",
        "bathrooms": "1",
        "floor": "2",
        "has_elevator": True,
        "city": "Roma",
        "macrozone": "Test",
        "microzone": "Test",
        "latitude": 41.9,
        "longitude": 12.5,
        "contract": "rent",
        "scraped_at": "2026-09-08T14:00:00+00:00",
    }
    row.update(overrides)
    return row


def test_clean_row_happy_path_computes_target():
    tr = ListingTransformer()
    cleaned, reason = tr.clean_row(_base_row(latitude=41.9052, longitude=12.4859))
    assert reason is None
    assert cleaned is not None
    assert cleaned["price_per_m2_monthly"] == 20.0
    assert cleaned["surface_m2"] == 60.0
    assert cleaned["rooms"] == 2
    assert cleaned["distance_from_center_km"] is not None
    assert cleaned["distance_from_center_km"] < 3.0


def test_clean_row_parses_rooms_plus_suffix():
    tr = ListingTransformer()
    cleaned, reason = tr.clean_row(_base_row(rooms="5+"))
    assert reason is None
    assert cleaned is not None
    assert cleaned["rooms"] == 5


@pytest.mark.parametrize(
    "overrides, expected_reason",
    [
        ({"price_eur_month": None}, "missing_price"),
        ({"surface_m2": None}, "missing_surface"),
        ({"rooms": None}, "missing_rooms"),
        ({"rooms": "n/a"}, "missing_rooms"),
        ({"surface_m2": "500 m²"}, "surface_out_of_bounds"),
        ({"price_eur_month": 20000}, "price_out_of_bounds"),
        ({"title": "Palazzo storico centro"}, "non_apartment_title"),
    ],
)
def test_clean_row_drop_reasons(overrides, expected_reason):
    tr = ListingTransformer()
    cleaned, reason = tr.clean_row(_base_row(**overrides))
    assert cleaned is None
    assert reason == expected_reason


# ---------------------------------------------------------------------------
# distance_from_center_km / haversine
# ---------------------------------------------------------------------------


def test_haversine_zero_for_same_point():
    assert ListingTransformer.haversine_km(41.9, 12.5, 41.9, 12.5) == 0.0


def test_distance_from_center_none_without_coords():
    tr = ListingTransformer()
    assert tr.distance_from_center_km(None, 12.5) is None
    assert tr.distance_from_center_km(41.9, None) is None


def test_distance_infernetto_farther_than_centro():
    tr = ListingTransformer()
    d_centro = tr.distance_from_center_km(41.9052, 12.4859)
    d_far = tr.distance_from_center_km(41.7517, 12.3724)
    assert d_centro is not None and d_far is not None
    assert d_far > d_centro
    assert 15.0 < d_far < 30.0
