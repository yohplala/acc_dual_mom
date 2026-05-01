"""Tests for correlation matrix + group identification on synthetic data."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from pea_momentum.correlations import (
    CorrelationMatrix,
    best_in_group,
    compute_correlation_matrix,
    find_groups,
)


def _synthetic_long(asset_returns: dict[str, np.ndarray], start: date) -> pl.DataFrame:
    """Build a long-format prices DataFrame from per-asset daily returns."""
    rows = []
    for asset_id, rets in asset_returns.items():
        prices = 100.0 * np.cumprod(1.0 + rets)
        for i, p in enumerate(prices):
            rows.append({"date": start + timedelta(days=i), "asset_id": asset_id, "close": p})
    return pl.DataFrame(rows).cast({"date": pl.Date, "asset_id": pl.Utf8, "close": pl.Float64})


class TestCorrelationMatrix:
    def test_perfectly_correlated_assets(self) -> None:
        rng = np.random.default_rng(42)
        rets = rng.normal(0, 0.01, 300)
        prices = _synthetic_long({"a": rets, "b": rets, "c": rets}, date(2024, 1, 1))
        cm = compute_correlation_matrix(prices, ["a", "b", "c"], window_days=200)
        assert sorted(cm.asset_ids) == ["a", "b", "c"]
        # All pairs should be near 1.0
        n = len(cm.asset_ids)
        for i in range(n):
            for j in range(n):
                assert cm.matrix[i, j] == pytest.approx(1.0, abs=1e-9)

    def test_uncorrelated_assets(self) -> None:
        rng = np.random.default_rng(42)
        rets_a = rng.normal(0, 0.01, 1000)
        rets_b = rng.normal(0, 0.01, 1000)  # independent
        prices = _synthetic_long({"a": rets_a, "b": rets_b}, date(2024, 1, 1))
        cm = compute_correlation_matrix(prices, ["a", "b"], window_days=900)
        # Correlation should be near zero
        assert abs(cm.matrix[0, 1]) < 0.15

    def test_inversely_correlated_assets(self) -> None:
        rng = np.random.default_rng(42)
        rets = rng.normal(0, 0.01, 500)
        prices = _synthetic_long({"a": rets, "b": -rets}, date(2024, 1, 1))
        cm = compute_correlation_matrix(prices, ["a", "b"], window_days=400)
        assert cm.matrix[0, 1] == pytest.approx(-1.0, abs=1e-9)

    def test_empty_input(self) -> None:
        empty = pl.DataFrame(schema={"date": pl.Date, "asset_id": pl.Utf8, "close": pl.Float64})
        cm = compute_correlation_matrix(empty, ["a"], window_days=100)
        assert cm.asset_ids == []
        assert cm.matrix.shape == (0, 0)


class TestFindGroups:
    def test_single_group_three_assets(self) -> None:
        cm = CorrelationMatrix(
            asset_ids=["a", "b", "c"],
            matrix=np.array([[1.0, 0.95, 0.90], [0.95, 1.0, 0.92], [0.90, 0.92, 1.0]]),
            window_days=252,
        )
        groups = find_groups(cm, threshold=0.85)
        assert len(groups) == 1
        assert sorted(groups[0]) == ["a", "b", "c"]

    def test_two_separate_groups(self) -> None:
        cm = CorrelationMatrix(
            asset_ids=["a", "b", "c", "d"],
            matrix=np.array(
                [
                    [1.0, 0.95, 0.10, 0.05],
                    [0.95, 1.0, 0.05, 0.08],
                    [0.10, 0.05, 1.0, 0.92],
                    [0.05, 0.08, 0.92, 1.0],
                ]
            ),
            window_days=252,
        )
        groups = find_groups(cm, threshold=0.85)
        assert len(groups) == 2
        # Largest group first
        assert sorted(groups[0]) == ["a", "b"]
        assert sorted(groups[1]) == ["c", "d"]

    def test_chained_grouping_via_transitivity(self) -> None:
        # a-b 0.90, b-c 0.90, a-c 0.50 → all three end up in one group
        cm = CorrelationMatrix(
            asset_ids=["a", "b", "c"],
            matrix=np.array([[1.0, 0.90, 0.50], [0.90, 1.0, 0.90], [0.50, 0.90, 1.0]]),
            window_days=252,
        )
        groups = find_groups(cm, threshold=0.85)
        assert len(groups) == 1
        assert sorted(groups[0]) == ["a", "b", "c"]

    def test_singletons_returned(self) -> None:
        cm = CorrelationMatrix(
            asset_ids=["a", "b"],
            matrix=np.array([[1.0, 0.10], [0.10, 1.0]]),
            window_days=252,
        )
        groups = find_groups(cm, threshold=0.85)
        # Two singleton groups
        assert len(groups) == 2
        assert all(len(g) == 1 for g in groups)


class TestBestInGroup:
    def test_picks_highest_score(self) -> None:
        # a: +20% over window, TER 0.20 → score = ~1.0
        # b: +10% over window, TER 0.10 → score = ~1.0
        # c: +30% over window, TER 0.10 → score = ~3.0 (winner)
        rng = np.random.default_rng(1)
        rets_a = rng.normal(0.20 / 252, 0.0001, 252)
        rets_b = rng.normal(0.10 / 252, 0.0001, 252)
        rets_c = rng.normal(0.30 / 252, 0.0001, 252)
        prices = _synthetic_long({"a": rets_a, "b": rets_b, "c": rets_c}, date(2024, 1, 1))

        rep = best_in_group(
            ["a", "b", "c"], prices, ter_pct_by_id={"a": 0.20, "b": 0.10, "c": 0.10}
        )
        assert rep.representative == "c"

    def test_singleton_group(self) -> None:
        rng = np.random.default_rng(1)
        prices = _synthetic_long({"a": rng.normal(0, 0.01, 252)}, date(2024, 1, 1))
        rep = best_in_group(["a"], prices, ter_pct_by_id={"a": 0.10})
        assert rep.representative == "a"
        assert rep.group == ["a"]
