"""Tests for rolling raw / feature-stamp prune."""

from __future__ import annotations

from pathlib import Path

from etl.prune_raw import prune_feature_stamps, prune_raw


def test_prune_raw_keeps_newest(tmp_path: Path):
    raw = tmp_path / "raw"
    for name in (
        "immobiliare_roma_20260901T000000Z",
        "immobiliare_roma_20260902T000000Z",
        "immobiliare_roma_20260903T000000Z",
    ):
        d = raw / name
        d.mkdir(parents=True)
        (d / "listings.jsonl").write_text("{}\n", encoding="utf-8")
        (d / "page_001.html").write_text("<html/>", encoding="utf-8")

    kept = prune_raw(raw_dir=raw, keep=1)
    assert [p.name for p in kept] == ["immobiliare_roma_20260903T000000Z"]
    assert not (raw / "immobiliare_roma_20260901T000000Z").exists()
    assert not list(raw.glob("**/*.html"))


def test_prune_feature_stamps(tmp_path: Path):
    (tmp_path / "features_latest.jsonl").write_text("a\n", encoding="utf-8")
    (tmp_path / "features_20260910T000000Z.jsonl").write_text("b\n", encoding="utf-8")
    n = prune_feature_stamps(processed_dir=tmp_path)
    assert n == 1
    assert (tmp_path / "features_latest.jsonl").is_file()
    assert not (tmp_path / "features_20260910T000000Z.jsonl").exists()
