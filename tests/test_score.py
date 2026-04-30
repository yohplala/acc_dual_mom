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
    cfg = Scoring(lookbacks_days=(21,), aggregation="median")
    with pytest.raises(ValueError, match="aggregation=mean"):
        score_at(df, asset_ids=[], as_of=date.today(), cfg=cfg)
