"""Backtest engine: vectorised polars compounding, sequential weight transitions.

Convention:
- Rebalance day R is a Sunday.
- Signal date S = R - 2d (Friday close used for scores).
- Fill date F = R + 1d (Monday close used as execution price).
- During day F, the OLD weights are held; at the close of F we transition to
  NEW weights and pay transaction cost on the turnover. From day F+1 onward
  the new weights apply.
- Costs: per_trade_pct is shared one-way (broker fee). Each asset can carry
  an `est_spread_bps` (estimated round-trip bid-ask spread); half is added
  per traded notional. Per-asset transition cost = |delta_w| * (per_trade_pct +
  est_spread_bps / 2) / 100. The total cost is summed across assets and
  charged as a negative return on the fill day.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date

import polars as pl

from . import stitching
from .allocate import CASH_KEY, allocate
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
    # Visibility into rebalance attrition. A degenerate strategy that ends
    # up with `len(rebalances) == 0` may have many fill_skips (rebalance
    # day past last available trading day) or signal_skips (no asset has
    # enough lookback yet); the CLI surfaces this so silent zero-rebalance
    # runs don't go unnoticed.
    n_fill_skips: int = 0
    n_signal_skips: int = 0


_EMPTY_EQUITY_SCHEMA = {
    "date": pl.Date,
    "equity": pl.Float64,
    "daily_return": pl.Float64,
}


def _empty_result(strategy_name: str) -> BacktestResult:
    """Empty result with no rebalances — used when there's no usable data."""
    return BacktestResult(
        strategy_name=strategy_name,
        equity=pl.DataFrame(schema=_EMPTY_EQUITY_SCHEMA),
        rebalances=[],
    )


def run(
    prices_long: pl.DataFrame,
    strategy: Strategy,
    config: Config,
    start: date | None = None,
    end: date | None = None,
) -> BacktestResult:
    asset_ids = list(strategy.asset_ids)
    safe_id = config.safe_asset.id
    # Residual-holder rule: rounding shortfalls and the "no candidate passed
    # the >0 filter" fallback go to the safe asset *if it is listed* in the
    # strategy's universe (so the residual earns €STR yield), otherwise to
    # CASH_KEY (a 0%-return placeholder).
    residual_holder = safe_id if safe_id in asset_ids else CASH_KEY

    # Defensive scrub: even if upstream prices.parquet has bad data
    # (round-trip spikes from yfinance bad days, or sustained-flat
    # forward-fill artefacts from older fetch versions), null those
    # closes here. Skipping this used to produce phantom equity-curve
    # spikes whose return-cancellation only worked at single-asset
    # weight=1; multi-asset portfolios accumulated permanent gains
    # from the spike+recovery pair.
    prices_long = stitching.scrub_long_format(prices_long)
    prices_long = prices_long.sort(["asset_id", "date"]).with_columns(
        close=pl.col("close").forward_fill().over("asset_id"),
    )

    available_ids = set(prices_long.get_column("asset_id").unique().to_list())
    wide = (
        prices_wide(prices_long, asset_ids)
        .sort("date")
        .with_columns([pl.col(c).forward_fill() for c in asset_ids if c in available_ids])
    )
    if start is not None:
        wide = wide.filter(pl.col("date") >= start)
    if end is not None:
        wide = wide.filter(pl.col("date") <= end)
    relevant_cols = [c for c in asset_ids if c in available_ids]
    if relevant_cols:
        # Drop rows where every listed asset is null (no data anywhere yet).
        wide = wide.filter(pl.any_horizontal([pl.col(c).is_not_null() for c in relevant_cols]))

    if strategy.mode == "buy_and_hold":
        return _run_buy_and_hold(wide, strategy, asset_ids, safe_id)

    if wide.is_empty():
        log.warning("backtest %s: no usable price data", strategy.name)
        return _empty_result(strategy.name)

    dates = wide.get_column("date").to_list()
    cost_pct = config.shared.costs.per_trade_pct / 100.0
    # Per-asset half-spread (bps → fraction). The safe asset's spread is 0
    # (€STR synthetic, no actual trade); CASH residuals trade at zero cost too.
    half_spread_by_id: dict[str, float] = {
        a.id: (a.est_spread_bps / 2.0) / 10_000.0 for a in config.assets
    }
    half_spread_by_id[safe_id] = 0.0
    half_spread_by_id[CASH_KEY] = 0.0
    scoring = strategy.effective_scoring(config.shared.scoring)
    date_set = set(dates)

    rebalances: list[Rebalance] = []
    fills: list[tuple[date, dict[str, float], float]] = []
    prev_weights: dict[str, float] = {residual_holder: 1.0}
    n_fill_skips = 0
    n_signal_skips = 0

    for r_day in rebalance_dates(strategy, dates[0], dates[-1]):
        s_day = signal_date(r_day)
        f_day = fill_date(r_day)
        if f_day not in date_set:
            n_fill_skips += 1
            continue
        scores = score_at(prices_long, asset_ids, s_day, scoring)
        if not scores:
            n_signal_skips += 1
            continue
        new_w = allocate(
            scores=scores,
            top_n=strategy.top_n,
            alloc=config.shared.allocation,
            flt=config.shared.filter,
            rule_override=strategy.allocation_rule,
            residual_holder=residual_holder,
        )
        turnover = _turnover(prev_weights, new_w)
        cost = _transition_cost(prev_weights, new_w, cost_pct, half_spread_by_id)
        rebalances.append(
            Rebalance(
                rebalance_date=r_day,
                signal_date=s_day,
                fill_date=f_day,
                scores=scores,
                weights=new_w,
                turnover=turnover,
                cost=cost,
            )
        )
        fills.append((f_day, new_w, cost))
        prev_weights = new_w

    # Compounding kernel needs every asset id that may appear in weights —
    # listed assets + the residual holder (CASH if safe wasn't listed). The
    # CASH placeholder has no price series; inject a synthetic constant
    # column so its daily return is exactly 0%.
    compound_ids = list(asset_ids)
    compound_wide = wide
    if residual_holder == CASH_KEY:
        compound_ids.append(CASH_KEY)
        compound_wide = compound_wide.with_columns(pl.lit(1.0).alias(CASH_KEY))
    equity_df = _compound(
        compound_wide, compound_ids, init_weights={residual_holder: 1.0}, fills=fills
    )
    return BacktestResult(
        strategy_name=strategy.name,
        equity=equity_df,
        rebalances=rebalances,
        n_fill_skips=n_fill_skips,
        n_signal_skips=n_signal_skips,
    )


def _turnover(prev: dict[str, float], new: dict[str, float]) -> float:
    """L1 distance between two weight dicts. Safe-asset rebalancing also counts."""
    keys = set(prev) | set(new)
    return sum(abs(new.get(k, 0.0) - prev.get(k, 0.0)) for k in keys)


def _transition_cost(
    prev: dict[str, float],
    new: dict[str, float],
    per_trade_pct_frac: float,
    half_spread_by_id: dict[str, float],
) -> float:
    """Per-asset rebalance cost = sum_i |delta_w_i| * (broker_fee + half_spread_i)."""
    keys = set(prev) | set(new)
    cost = 0.0
    for k in keys:
        delta = abs(new.get(k, 0.0) - prev.get(k, 0.0))
        cost += delta * (per_trade_pct_frac + half_spread_by_id.get(k, 0.0))
    return cost


def _run_buy_and_hold(
    wide: pl.DataFrame,
    strategy: Strategy,
    asset_ids: list[str],
    safe_id: str,
) -> BacktestResult:
    """Buy-and-hold benchmark. Default = equal-weight across `asset_ids`;
    when `strategy.static_weights` is set (e.g. 60/40), use those instead.
    Static weights may reference the safe asset id. No rebalances, no
    transaction costs."""
    if strategy.static_weights is not None:
        weights = dict(strategy.static_weights)
        asset_cols = [a for a in weights if a in wide.columns]
        if len(asset_cols) != len(weights):
            missing = [a for a in weights if a not in wide.columns]
            log.warning(
                "buy_and_hold %s: static_weights references unavailable assets %s",
                strategy.name,
                missing,
            )
    else:
        asset_cols = [c for c in asset_ids if c in wide.columns]
        if not asset_cols:
            return _empty_result(strategy.name)
        n = len(asset_cols)
        weights = {a: 1.0 / n for a in asset_cols}

    if not asset_cols:
        return _empty_result(strategy.name)
    # Drop rows where any held asset has no price — guarantees the equity
    # curve starts on the latest-listed sleeve's first available date.
    wide = wide.drop_nulls(subset=asset_cols)
    if wide.is_empty():
        return _empty_result(strategy.name)

    dates = wide.get_column("date").to_list()
    equity_df = _compound(wide, asset_cols, init_weights=weights, fills=[])
    _ = safe_id  # unused for now; kept for symmetry with rotation entry-point

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
    return BacktestResult(strategy_name=strategy.name, equity=equity_df, rebalances=rebalances)


def _compound(
    wide: pl.DataFrame,
    asset_cols: list[str],
    init_weights: dict[str, float],
    fills: list[tuple[date, dict[str, float], float]],
) -> pl.DataFrame:
    """Polars-vectorised daily compounding kernel shared by rotation + buy-and-hold.

    `wide` is `[date, *asset_cols]`. `init_weights` is the portfolio at `dates[0]`.
    Each `fills` entry is `(fill_date, new_weights, cost_on_fill_day)` — the new
    weights become effective on the next trading day after `fill_date`, and the
    cost is charged as a negative return on `fill_date` itself.

    Returns `[date, equity, daily_return]`. Equity[0] == 1.0; subsequent values
    compound the strategy's net daily return.
    """
    if wide.is_empty():
        return pl.DataFrame(schema=_EMPTY_EQUITY_SCHEMA)

    dates = wide.get_column("date").to_list()
    date_idx = {d: i for i, d in enumerate(dates)}

    # Sparse weights: row at dates[0] = init, plus one row per fill (effective
    # the trading day AFTER the fill_date, since the close on the fill_date
    # itself is the execution price for the OLD weights).
    init_row: dict[str, object] = {"date": dates[0]}
    for c in asset_cols:
        init_row[c] = float(init_weights.get(c, 0.0))
    sparse_rows: list[dict[str, object]] = [init_row]
    for f_day, new_w, _cost in fills:
        idx = date_idx.get(f_day)
        if idx is None or idx + 1 >= len(dates):
            continue  # fill on or past last trading day — no future returns to weight
        row: dict[str, object] = {"date": dates[idx + 1]}
        for c in asset_cols:
            row[c] = float(new_w.get(c, 0.0))
        sparse_rows.append(row)

    sparse_schema: dict[str, type[pl.DataType]] = {"date": pl.Date}
    for c in asset_cols:
        sparse_schema[c] = pl.Float64
    sparse = (
        pl.DataFrame(sparse_rows, schema=sparse_schema)
        .sort("date")
        .unique(subset=["date"], keep="last")
    )

    weights_per_day = (
        pl.DataFrame({"date": dates}, schema={"date": pl.Date})
        .join(sparse, on="date", how="left")
        .with_columns(*[pl.col(c).forward_fill() for c in asset_cols])
    )

    if fills:
        cost_sparse = pl.DataFrame(
            {"date": [f[0] for f in fills], "cost": [float(f[2]) for f in fills]},
            schema={"date": pl.Date, "cost": pl.Float64},
        )
        cost_per_day = (
            pl.DataFrame({"date": dates}, schema={"date": pl.Date})
            .join(cost_sparse, on="date", how="left")
            .with_columns(pl.col("cost").fill_null(0.0))
        )
    else:
        cost_per_day = pl.DataFrame(
            {"date": dates, "cost": [0.0] * len(dates)},
            schema={"date": pl.Date, "cost": pl.Float64},
        )

    rets = wide.select(
        pl.col("date"),
        *[(pl.col(c) / pl.col(c).shift(1) - 1.0).fill_null(0.0).alias(c) for c in asset_cols],
    )

    w_suffix = "__w"
    weights_renamed = weights_per_day.rename({c: c + w_suffix for c in asset_cols})
    df = rets.join(weights_renamed, on="date").join(cost_per_day, on="date").sort("date")

    gross_expr = pl.sum_horizontal([pl.col(c) * pl.col(c + w_suffix) for c in asset_cols])
    return (
        df.with_columns(daily_return=gross_expr - pl.col("cost"))
        .with_columns(equity=(1.0 + pl.col("daily_return")).cum_prod())
        .select(["date", "equity", "daily_return"])
    )


def rebalances_to_json(rebalances: list[Rebalance]) -> str:
    return json.dumps(
        [
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
        ],
        indent=2,
        default=str,
    )


def rebalances_from_json(text: str) -> list[Rebalance]:
    """Strict deserializer mirroring `rebalances_to_json`. Missing keys raise
    KeyError — they would indicate a corrupted artifact, not a legacy schema."""
    raw = json.loads(text)
    return [
        Rebalance(
            rebalance_date=date.fromisoformat(r["rebalance_date"]),
            signal_date=date.fromisoformat(r["signal_date"]),
            fill_date=date.fromisoformat(r["fill_date"]),
            scores=r["scores"],
            weights=r["weights"],
            turnover=float(r["turnover"]),
            cost=float(r["cost"]),
        )
        for r in raw
    ]
