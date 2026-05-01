"""Backtest engine: vectorized prices, sequential weight transitions.

Convention:
- Rebalance day R is a Sunday.
- Signal date S = R - 2d (Friday close used for scores).
- Fill date F = R + 1d (Monday close used as execution price).
- During day F, the OLD weights are held; at the close of F we transition to
  NEW weights and pay transaction cost on the turnover. From day F+1 onward
  the new weights apply.
- Costs: `per_trade_pct` is one-way; total cost = turnover_l1 * per_trade_pct.
  Charged as a negative return on the fill day.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date

import polars as pl

from .allocate import SAFE_ASSET_KEY, allocate
from .schedule import fill_date, rebalance_dates, signal_date
from .score import score_at
from .store import prices_wide
from .universe import Config, Strategy

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Rebalance:
    rebalance_date: date
    signal_date: date
    fill_date: date
    scores: dict[str, float]
    weights: dict[str, float]
    turnover: float
    cost: float


@dataclass(frozen=True, slots=True)
class BacktestResult:
    strategy_name: str
    equity: pl.DataFrame  # [date, equity, daily_return]
    rebalances: list[Rebalance]
    weight_history: pl.DataFrame  # [date, asset_id, weight]


def run(
    prices_long: pl.DataFrame,
    strategy: Strategy,
    config: Config,
    start: date | None = None,
    end: date | None = None,
) -> BacktestResult:
    asset_ids = list(strategy.asset_ids)
    safe_id = config.safe_asset.id
    all_ids = [*asset_ids, safe_id]

    wide = (
        prices_wide(prices_long, all_ids)
        .sort("date")
        .with_columns(
            [
                pl.col(c).forward_fill()
                for c in all_ids
                if c in prices_long.get_column("asset_id").unique().to_list()
            ]
        )
    )
    if start is not None:
        wide = wide.filter(pl.col("date") >= start)
    if end is not None:
        wide = wide.filter(pl.col("date") <= end)
    wide = wide.drop_nulls(subset=[safe_id])  # need at least the safe asset

    if strategy.mode == "buy_and_hold":
        return _run_buy_and_hold(wide, strategy, asset_ids, safe_id)

    if wide.is_empty():
        log.warning("backtest %s: no usable price data", strategy.name)
        return BacktestResult(
            strategy_name=strategy.name,
            equity=pl.DataFrame(
                schema={"date": pl.Date, "equity": pl.Float64, "daily_return": pl.Float64}
            ),
            rebalances=[],
            weight_history=pl.DataFrame(
                schema={"date": pl.Date, "asset_id": pl.Utf8, "weight": pl.Float64}
            ),
        )

    dates = wide.get_column("date").to_list()
    cost_pct = config.shared.costs.per_trade_pct / 100.0

    # Schedule: fill_date -> new_weights
    rebalances: list[Rebalance] = []
    fill_to_weights: dict[date, dict[str, float]] = {}
    prev_weights: dict[str, float] = {SAFE_ASSET_KEY: 1.0}
    safe_score_id = safe_id

    for r_day in rebalance_dates(strategy, dates[0], dates[-1]):
        s_day = signal_date(r_day)
        f_day = fill_date(r_day)
        if f_day not in dates:
            continue  # fill day outside trading window
        scores = score_at(prices_long, asset_ids, s_day, config.shared.scoring)
        safe_scores = score_at(prices_long, [safe_score_id], s_day, config.shared.scoring)
        safe_score = safe_scores.get(safe_score_id, 0.0)
        if not scores:
            continue
        new_w = allocate(
            scores=scores,
            safe_score=safe_score,
            top_n=strategy.top_n,
            alloc=config.shared.allocation,
            flt=config.shared.filter,
        )
        # Map "safe" sentinel from allocate() to the actual safe asset id
        new_w_mapped = {(safe_id if a == SAFE_ASSET_KEY else a): w for a, w in new_w.items()}
        turnover = _turnover(prev_weights, new_w_mapped, safe_id)
        rebalances.append(
            Rebalance(
                rebalance_date=r_day,
                signal_date=s_day,
                fill_date=f_day,
                scores=scores,
                weights=new_w_mapped,
                turnover=turnover,
                cost=turnover * cost_pct,
            )
        )
        fill_to_weights[f_day] = new_w_mapped
        prev_weights = new_w_mapped

    # Per-day effective weights and per-day cost
    effective: list[dict[str, float]] = []
    cost_on_day: list[float] = []
    held: dict[str, float] = {safe_id: 1.0}
    prev_w_for_cost: dict[str, float] = {safe_id: 1.0}

    for d in dates:
        effective.append(dict(held))
        c = 0.0
        if d in fill_to_weights:
            new_w = fill_to_weights[d]
            c = _turnover(prev_w_for_cost, new_w, safe_id) * cost_pct
            held = dict(new_w)
            prev_w_for_cost = dict(new_w)
        cost_on_day.append(c)

    # Daily asset returns
    asset_cols = [c for c in wide.columns if c != "date"]
    rets_wide = wide.select(
        pl.col("date"),
        *[(pl.col(c) / pl.col(c).shift(1) - 1.0).fill_null(0.0).alias(c) for c in asset_cols],
    )
    rets_rows = rets_wide.iter_rows(named=True)
    next(rets_rows)  # skip first (zero) row, weights[0] applies but no return

    equity_values: list[float] = [1.0]
    daily_returns: list[float] = [0.0]
    for i, row in enumerate(rets_rows, start=1):
        w = effective[i]
        gross = sum(w.get(a, 0.0) * row.get(a, 0.0) for a in asset_cols)
        net = gross - cost_on_day[i]
        daily_returns.append(net)
        equity_values.append(equity_values[-1] * (1.0 + net))

    equity_df = pl.DataFrame(
        {
            "date": dates,
            "equity": equity_values,
            "daily_return": daily_returns,
        }
    )

    # Long-format weight history (one row per date+asset combination)
    history_rows: list[dict[str, object]] = []
    for d, w in zip(dates, effective, strict=True):
        for a, weight in w.items():
            if weight > 0:
                history_rows.append({"date": d, "asset_id": a, "weight": weight})
    weight_history = (
        pl.DataFrame(
            history_rows, schema={"date": pl.Date, "asset_id": pl.Utf8, "weight": pl.Float64}
        )
        if history_rows
        else pl.DataFrame(schema={"date": pl.Date, "asset_id": pl.Utf8, "weight": pl.Float64})
    )

    return BacktestResult(
        strategy_name=strategy.name,
        equity=equity_df,
        rebalances=rebalances,
        weight_history=weight_history,
    )


def _turnover(prev: dict[str, float], new: dict[str, float], safe_id: str) -> float:
    """L1 distance between two weight dicts. Safe-asset rebalancing also counts."""
    keys = set(prev) | set(new)
    return sum(abs(new.get(k, 0.0) - prev.get(k, 0.0)) for k in keys)


def _run_buy_and_hold(
    wide: pl.DataFrame,
    strategy: Strategy,
    asset_ids: list[str],
    safe_id: str,
) -> BacktestResult:
    """Equal-weight buy-and-hold across `asset_ids` from the first available
    date. No rebalances, no transaction costs — pure benchmark."""
    asset_cols = [c for c in asset_ids if c in wide.columns]
    if not asset_cols:
        return BacktestResult(
            strategy_name=strategy.name,
            equity=pl.DataFrame(
                schema={"date": pl.Date, "equity": pl.Float64, "daily_return": pl.Float64}
            ),
            rebalances=[],
            weight_history=pl.DataFrame(
                schema={"date": pl.Date, "asset_id": pl.Utf8, "weight": pl.Float64}
            ),
        )

    wide = wide.drop_nulls(subset=asset_cols)
    if wide.is_empty():
        return BacktestResult(
            strategy_name=strategy.name,
            equity=pl.DataFrame(
                schema={"date": pl.Date, "equity": pl.Float64, "daily_return": pl.Float64}
            ),
            rebalances=[],
            weight_history=pl.DataFrame(
                schema={"date": pl.Date, "asset_id": pl.Utf8, "weight": pl.Float64}
            ),
        )

    dates = wide.get_column("date").to_list()
    n = len(asset_cols)
    weight_per = 1.0 / n
    weights = {a: weight_per for a in asset_cols}

    rets_wide = wide.select(
        pl.col("date"),
        *[(pl.col(c) / pl.col(c).shift(1) - 1.0).fill_null(0.0).alias(c) for c in asset_cols],
    )
    rows = list(rets_wide.iter_rows(named=True))

    equity_values: list[float] = [1.0]
    daily_returns: list[float] = [0.0]
    for i in range(1, len(rows)):
        gross = sum(weight_per * rows[i].get(a, 0.0) for a in asset_cols)
        daily_returns.append(gross)
        equity_values.append(equity_values[-1] * (1.0 + gross))

    equity_df = pl.DataFrame(
        {"date": dates, "equity": equity_values, "daily_return": daily_returns}
    )

    # Single synthetic rebalance at start so the dashboard shows the static
    # weights in "New allocation". turnover and cost are zero by definition.
    rebalances = [
        Rebalance(
            rebalance_date=dates[0],
            signal_date=dates[0],
            fill_date=dates[0],
            scores={},
            weights=weights,
            turnover=0.0,
            cost=0.0,
        )
    ]

    history_rows = [{"date": d, "asset_id": a, "weight": weight_per} for d in dates for a in asset_cols]
    weight_history = pl.DataFrame(
        history_rows, schema={"date": pl.Date, "asset_id": pl.Utf8, "weight": pl.Float64}
    )

    return BacktestResult(
        strategy_name=strategy.name,
        equity=equity_df,
        rebalances=rebalances,
        weight_history=weight_history,
    )


def rebalances_to_dicts(rebalances: list[Rebalance]) -> list[dict[str, object]]:
    return [
        {
            "rebalance_date": r.rebalance_date.isoformat(),
            "signal_date": r.signal_date.isoformat(),
            "fill_date": r.fill_date.isoformat(),
            "scores": r.scores,
            "weights": r.weights,
            "turnover": r.turnover,
            "cost": r.cost,
        }
        for r in rebalances
    ]


def rebalances_to_json(rebalances: list[Rebalance]) -> str:
    return json.dumps(rebalances_to_dicts(rebalances), indent=2, default=str)
