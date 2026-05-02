"""Performance metrics on a daily equity curve.

All functions take a polars DataFrame `[date, equity]` (sorted ascending) and
return a single float or a dict. Returns are derived from the equity column.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import numpy as np
import polars as pl

from .correlations import pairwise_corrcoef

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True, slots=True)
class Metrics:
    cagr: float
    vol_ann: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    hit_rate: float
    final_equity: float
    n_days: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute(equity: pl.DataFrame) -> Metrics:
    """Sharpe / Sortino are computed on raw daily returns (rf=0). The dual-
    momentum framework already absorbs the risk-free rate at the strategy
    level via the absolute-momentum filter against the safe asset, so
    excess-vs-rf adjustment here would double-count it."""
    if equity.is_empty():
        return Metrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0)

    eq = equity.sort("date")
    values = eq.get_column("equity").to_list()
    n = len(values)
    if n < 2:
        return Metrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, values[0], n)

    daily_returns = [values[i] / values[i - 1] - 1.0 for i in range(1, n)]
    years = (eq.get_column("date")[-1] - eq.get_column("date")[0]).days / 365.25
    cagr = (values[-1] / values[0]) ** (1.0 / years) - 1.0 if years > 0 else 0.0

    mean_r = sum(daily_returns) / len(daily_returns)
    var_r = sum((r - mean_r) ** 2 for r in daily_returns) / max(1, len(daily_returns) - 1)
    vol_daily = math.sqrt(var_r)
    vol_ann = vol_daily * math.sqrt(TRADING_DAYS_PER_YEAR)

    sharpe = mean_r / vol_daily * math.sqrt(TRADING_DAYS_PER_YEAR) if vol_daily > 0 else 0.0

    downside = [r for r in daily_returns if r < 0]
    if downside:
        ds_var = sum(r * r for r in downside) / len(downside)
        ds_vol = math.sqrt(ds_var)
        sortino = mean_r / ds_vol * math.sqrt(TRADING_DAYS_PER_YEAR) if ds_vol > 0 else 0.0
    else:
        sortino = float("inf") if mean_r > 0 else 0.0

    peak = values[0]
    max_dd = 0.0
    for v in values:
        peak = max(peak, v)
        dd = v / peak - 1.0
        max_dd = min(max_dd, dd)

    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0
    hit_rate = sum(1 for r in daily_returns if r > 0) / len(daily_returns)

    return Metrics(
        cagr=cagr,
        vol_ann=vol_ann,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        calmar=calmar,
        hit_rate=hit_rate,
        final_equity=values[-1],
        n_days=n,
    )


def drawdown_series(equity: pl.DataFrame) -> pl.DataFrame:
    return (
        equity.sort("date")
        .with_columns(
            pl.col("equity").cum_max().alias("_peak"),
        )
        .with_columns(
            drawdown=(pl.col("equity") / pl.col("_peak") - 1.0),
        )
        .select(["date", "drawdown"])
    )


def turnover_per_year(
    equity: pl.DataFrame,
    turnovers: list[float],
) -> float | None:
    """Average annualised L1 turnover (sum across rebalances / years).

    Returns `None` when there are fewer than two rebalances or insufficient
    history. Higher numbers mean higher trading activity — directly drives
    transaction-cost drag at fixed `per_trade_pct`.
    """
    if not turnovers or equity.is_empty():
        return None
    eq = equity.sort("date")
    span = eq.get_column("date")[-1] - eq.get_column("date")[0]
    years = span.days / 365.25
    if years <= 0:
        return None
    return float(sum(turnovers) / years)


def rebalance_hit_rate(
    equity: pl.DataFrame,
    fill_dates: list[date],
) -> float | None:
    """Fraction of rebalance intervals where the strategy's equity grew.

    For each pair of consecutive fill dates `(F_i, F_{i+1})`, look up the
    equity values at those dates and check whether `equity(F_{i+1}) >
    equity(F_i)`. The final interval `(F_last, end_of_history)` is included.

    Per-rebalance is far more meaningful than per-day for a slow rotation
    strategy: a monthly strategy has ~120 rebalances over 10 years, so this
    is a real Bernoulli-trial sample size; the daily hit-rate is correlated
    with vol_ann more than skill.

    Returns `None` when there are fewer than two fill dates or none fall
    inside the equity-curve range.
    """
    if not fill_dates or equity.is_empty():
        return None
    eq = equity.sort("date")
    eq_dates = eq.get_column("date").to_list()
    eq_values = eq.get_column("equity").to_list()
    by_date: dict[date, float] = dict(zip(eq_dates, eq_values, strict=True))

    # Anchor points: each fill_date that exists in the equity curve, plus
    # the final equity-curve date so the last interval contributes too.
    anchors: list[float] = []
    for d in sorted(set(fill_dates)):
        if d in by_date:
            anchors.append(by_date[d])
    if not anchors:
        return None
    anchors.append(eq_values[-1])
    if len(anchors) < 2:
        return None

    wins = sum(1 for i in range(1, len(anchors)) if anchors[i] > anchors[i - 1])
    return wins / (len(anchors) - 1)


def avg_pairwise_correlation(
    prices_long: pl.DataFrame,
    asset_ids: list[str],
) -> float | None:
    """Average of off-diagonal pairwise correlations of daily returns over the
    full available history of `asset_ids`. Returns `None` if fewer than two
    assets have usable data.

    Lower values indicate a more decorrelated universe — better diversification
    potential, which momentum-rotation strategies thrive on. Typical ranges:

      < 0.40  well-diversified (e.g. equities + bonds + commodities)
      0.40-0.60  moderate (regional equities)
      0.60-0.75  tightly correlated (developed-market equities)
      > 0.75  near-redundant (sector ETFs within one region)
    """
    result = pairwise_corrcoef(prices_long, asset_ids, window_days=None)
    if result is None:
        return None
    _, corr = result
    n = corr.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(corr[mask].mean())
