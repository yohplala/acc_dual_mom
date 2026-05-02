"""Tests for ROC averaging score."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from pea_momentum.score import _score_series, score_at
from pea_momentum.universe import Scoring


def test_score_series_simple() -> None:
    closes = [100.0] * 130
    closes[-1] = 110.0
    closes[-22] = 100.0
    closes[-64] = 100.0
    closes[-127] = 100.0
    s = _score_series(closes, lookbacks=(21, 63, 126))
    assert s == pytest.approx(0.10, abs=1e-9)


def test_score_series_negative() -> None:
    closes = [100.0 - i * 0.1 for i in range(130)]
    s = _score_series(closes, lookbacks=(21, 63, 126))
    assert s is not None and s < 0


def test_score_series_insufficient_history() -> None:
    closes = [100.0] * 50
    assert _score_series(closes, lookbacks=(21, 63, 126)) is None


def test_score_series_empty() -> None:
    assert _score_series([], lookbacks=(21, 63, 126)) is None


def test_score_at_filters_assets() -> None:
    rows = []
    base = date(2024, 1, 1)
    for i in range(130):
        d = base + timedelta(days=i)
        rows.append({"date": d, "asset_id": "a", "close": 100.0 + i})
        rows.append({"date": d, "asset_id": "b", "close": 200.0 - i * 0.1})
    df = pl.DataFrame(rows).cast({"date": pl.Date, "asset_id": pl.Utf8, "close": pl.Float64})

    cfg = Scoring(lookbacks_days=(21, 63, 126), aggregation="mean")
    scores = score_at(df, asset_ids=["a", "b"], as_of=base + timedelta(days=129), cfg=cfg)
    assert set(scores) == {"a", "b"}
    assert scores["a"] > 0
    assert scores["b"] < 0


def test_score_at_unknown_aggregation_raises() -> None:
    df = pl.DataFrame(schema={"date": pl.Date, "asset_id": pl.Utf8, "close": pl.Float64})
    cfg = Scoring(lookbacks_days=(21,), aggregation="harmonic_mean")
    with pytest.raises(ValueError, match="Unsupported aggregation"):
        score_at(df, asset_ids=[], as_of=date.today(), cfg=cfg)


def test_score_series_median_robust_to_outlier() -> None:
    """Median ignores a single anomalous lookback; mean would pull toward it."""
    # Build a series where the 21-day lookback shows a -10% drop while the
    # 63-day and 126-day windows are flat. Mean would average to ~-3%; median
    # picks 0%.
    closes = [100.0] * 130
    closes[-22] = 100.0 / 0.90  # 21-day return = -10%
    s_mean = _score_series(closes, lookbacks=(21, 63, 126), aggregation="mean")
    s_median = _score_series(closes, lookbacks=(21, 63, 126), aggregation="median")
    assert s_mean is not None and s_mean < -0.02
    assert s_median == pytest.approx(0.0, abs=1e-9)


def test_score_series_min_pessimistic() -> None:
    closes = [100.0 + i * 0.1 for i in range(130)]
    s_min = _score_series(closes, lookbacks=(21, 63, 126), aggregation="min")
    s_mean = _score_series(closes, lookbacks=(21, 63, 126), aggregation="mean")
    assert s_min is not None and s_mean is not None
    assert s_min <= s_mean


def test_score_at_excludes_assets_with_insufficient_history() -> None:
    """Asset `b` only has 50 days of data — it can't satisfy the 126-day
    lookback so it must be dropped from the score dict, not silently scored
    on a partial mean."""
    base = date(2024, 1, 1)
    rows = []
    for i in range(130):
        d = base + timedelta(days=i)
        rows.append({"date": d, "asset_id": "a", "close": 100.0 + i})
    for i in range(50):
        d = base + timedelta(days=80 + i)
        rows.append({"date": d, "asset_id": "b", "close": 100.0 + i})
    df = pl.DataFrame(rows).cast({"date": pl.Date, "asset_id": pl.Utf8, "close": pl.Float64})
    scores = score_at(
        df,
        asset_ids=["a", "b"],
        as_of=base + timedelta(days=129),
        cfg=Scoring(lookbacks_days=(21, 63, 126), aggregation="mean"),
    )
    assert "a" in scores
    assert "b" not in scores  # not silently averaged on the 21- + 63-day pair


def test_score_at_median_aggregation_path() -> None:
    """End-to-end check that the median list.median() path returns the same
    result as _score_series.median for a controlled input."""
    base = date(2024, 1, 1)
    rows = []
    closes = [100.0] * 130
    closes[-22] = 100.0 / 0.90  # injects a 21-day -10% lookback
    for i, c in enumerate(closes):
        rows.append({"date": base + timedelta(days=i), "asset_id": "a", "close": c})
    df = pl.DataFrame(rows).cast({"date": pl.Date, "asset_id": pl.Utf8, "close": pl.Float64})
    scores = score_at(
        df,
        asset_ids=["a"],
        as_of=base + timedelta(days=129),
        cfg=Scoring(lookbacks_days=(21, 63, 126), aggregation="median"),
    )
    assert scores["a"] == pytest.approx(0.0, abs=1e-9)
