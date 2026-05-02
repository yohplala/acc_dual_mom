"""Tests for performance metrics: equity-curve aggregates + rebalance-level."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from pea_momentum.metrics import (
    compute,
    rebalance_hit_rate,
    turnover_per_year,
)


def _equity(dates: list[date], values: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"date": dates, "equity": values}).with_columns(
        pl.col("date").cast(pl.Date), pl.col("equity").cast(pl.Float64)
    )


def test_compute_handles_empty() -> None:
    m = compute(pl.DataFrame(schema={"date": pl.Date, "equity": pl.Float64}))
    assert m.cagr == 0.0
    assert m.n_days == 0


def test_compute_uptrend_positive_metrics() -> None:
    # 252 days, +0.04% per day → ~10% annual
    dates = [date(2023, 1, 1) + timedelta(days=i) for i in range(252)]
    values = [(1.0004) ** i for i in range(252)]
    m = compute(_equity(dates, values))
    assert m.cagr > 0.05
    assert m.sharpe > 0
    assert m.max_drawdown == pytest.approx(0.0, abs=1e-9)


class TestTurnoverPerYear:
    def test_none_when_no_rebalances(self) -> None:
        eq = _equity([date(2023, 1, 1), date(2024, 1, 1)], [1.0, 1.1])
        assert turnover_per_year(eq, []) is None

    def test_simple_annualisation(self) -> None:
        # 1-year span, 3 rebalances of L1=1.0 each → 3.0/yr
        dates = [date(2023, 1, 1), date(2024, 1, 1)]  # ~365.25 days
        eq = _equity(dates, [1.0, 1.1])
        assert turnover_per_year(eq, [1.0, 1.0, 1.0]) == pytest.approx(3.0, abs=0.01)

    def test_none_when_zero_year_span(self) -> None:
        eq = _equity([date(2023, 1, 1)], [1.0])
        assert turnover_per_year(eq, [1.0]) is None


class TestRebalanceHitRate:
    def test_all_winning_intervals(self) -> None:
        # Equity grows monotonically; every interval is a win.
        dates = [date(2023, 1, 1) + timedelta(days=i) for i in range(10)]
        values = [1.0 + 0.01 * i for i in range(10)]
        eq = _equity(dates, values)
        # Fill dates exist in eq; final interval (last fill → end) also wins.
        rate = rebalance_hit_rate(eq, [dates[1], dates[3], dates[5]])
        assert rate == pytest.approx(1.0)

    def test_mixed_outcomes(self) -> None:
        dates = [date(2023, 1, 1) + timedelta(days=i) for i in range(10)]
        # Equity oscillates: rebalances at days 1, 4, 7
        # eq at day 1 = 1.0, day 4 = 1.1, day 7 = 0.9, day 9 (end) = 1.05
        # → intervals: 1.0→1.1 win, 1.1→0.9 loss, 0.9→1.05 win  ⇒ 2/3
        values = [1.0, 1.0, 1.05, 1.08, 1.1, 1.0, 0.95, 0.9, 0.95, 1.05]
        eq = _equity(dates, values)
        rate = rebalance_hit_rate(eq, [dates[1], dates[4], dates[7]])
        assert rate == pytest.approx(2.0 / 3.0)

    def test_none_when_no_rebalances(self) -> None:
        eq = _equity([date(2023, 1, 1), date(2024, 1, 1)], [1.0, 1.1])
        assert rebalance_hit_rate(eq, []) is None
