"""Unit tests for FeatureBuilder (RF-04 features)."""

from __future__ import annotations

from etl.transform.features_phase import FeatureBuilder


def test_leave_one_out_zone_mean():
    rows = [
        {"macrozone": "A", "price_per_m2_monthly": 20.0},
        {"macrozone": "A", "price_per_m2_monthly": 30.0},
        {"macrozone": "A", "price_per_m2_monthly": 40.0},
    ]
    hist = FeatureBuilder._leave_one_out_zone_means(rows)
    assert hist == [35.0, 30.0, 25.0]


def test_single_row_zone_is_none():
    rows = [
        {"macrozone": "Solo", "price_per_m2_monthly": 22.0},
        {"macrozone": "A", "price_per_m2_monthly": 10.0},
        {"macrozone": "A", "price_per_m2_monthly": 20.0},
    ]
    hist = FeatureBuilder._leave_one_out_zone_means(rows)
    assert hist[0] is None
    assert hist[1] == 20.0
    assert hist[2] == 10.0


def test_missing_macrozone_is_none():
    rows = [
        {"macrozone": None, "price_per_m2_monthly": 15.0},
        {"macrozone": "", "price_per_m2_monthly": 16.0},
        {"macrozone": "A", "price_per_m2_monthly": 10.0},
        {"macrozone": "A", "price_per_m2_monthly": 20.0},
    ]
    hist = FeatureBuilder._leave_one_out_zone_means(rows)
    assert hist[0] is None
    assert hist[1] is None
    assert hist[2] == 20.0
    assert hist[3] == 10.0


def test_publication_month_season():
    month, season = FeatureBuilder.publication_month_season(
        "2026-09-08T14:11:52.456099+00:00"
    )
    assert month == 9
    assert season == "autumn"
    assert FeatureBuilder.publication_month_season(None) == (None, None)
    assert FeatureBuilder.publication_month_season("not-a-date") == (None, None)


def test_municipio_from_macrozone():
    assert FeatureBuilder.municipio_from_macrozone("Centro Storico") == "I"
    assert FeatureBuilder.municipio_from_macrozone("Eur, Torrino, Tintoretto") == "IX"
    assert FeatureBuilder.municipio_from_macrozone(None) is None
    assert FeatureBuilder.municipio_from_macrozone("Unknown Zone") is None


def test_enrich_appends_all_features(tmp_path):
    fb = FeatureBuilder(input_path=tmp_path / "x.jsonl", processed_dir=tmp_path)
    rows = [
        {
            "listing_id": 1,
            "macrozone": "Prati, Borgo, Mazzini, Delle Vittorie, Degli Eroi",
            "price_per_m2_monthly": 20.0,
            "rooms": 2,
            "scraped_at": "2026-07-01T10:00:00+00:00",
        },
        {
            "listing_id": 2,
            "macrozone": "Prati, Borgo, Mazzini, Delle Vittorie, Degli Eroi",
            "price_per_m2_monthly": 30.0,
            "rooms": 3,
            "scraped_at": "2026-07-01T10:00:00+00:00",
        },
    ]
    out = fb.enrich(rows)
    assert len(out) == 2
    assert out[0]["listing_id"] == 1
    assert out[0]["rooms"] == 2
    assert out[0]["area_price_per_m2_hist"] == 30.0
    assert out[1]["area_price_per_m2_hist"] == 20.0
    assert out[0]["publication_month"] == 7
    assert out[0]["publication_season"] == "summer"
    assert out[0]["municipio"] == "I"


def test_write_outputs_creates_latest(tmp_path):
    fb = FeatureBuilder(input_path=tmp_path / "in.jsonl", processed_dir=tmp_path)
    rows = [{"listing_id": 1, "area_price_per_m2_hist": 10.0}]
    out = fb.write_outputs(rows)
    assert out.is_file()
    assert (tmp_path / "features_latest.jsonl").is_file()
    assert "features_" in out.name
