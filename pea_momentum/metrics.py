"""Performance metrics on a daily equity curve.

All functions take a polars DataFrame `[date, equity]` (sorted ascending) and
return a single float or a dict. Returns are derived from the equity column.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import polars as pl

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


def compute(equity: pl.DataFrame, rf_annual: float = 0.0) -> Metrics:
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

    rf_daily = (1.0 + rf_annual) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    excess = [r - rf_daily for r in daily_returns]
    sharpe = (
        (sum(excess) / len(excess)) / vol_daily * math.sqrt(TRADING_DAYS_PER_YEAR)
        if vol_daily > 0
        else 0.0
    )

    downside = [r for r in excess if r < 0]
    if downside:
        ds_var = sum(r * r for r in downside) / len(downside)
        ds_vol = math.sqrt(ds_var)
        sortino = (
            (sum(excess) / len(excess)) / ds_vol * math.sqrt(TRADING_DAYS_PER_YEAR)
            if ds_vol > 0
            else 0.0
        )
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
    from .correlations import pairwise_corrcoef

    result = pairwise_corrcoef(prices_long, asset_ids, window_days=None)
    if result is None:
        return None
    _, corr = result
    n = corr.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(corr[mask].mean())
