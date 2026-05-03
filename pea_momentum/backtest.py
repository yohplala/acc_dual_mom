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
from datetime import date, timedelta

import polars as pl

from . import stitching
from .allocate import CASH_KEY, allocate
from .discover import assets_by_region
from .schedule import _SUNDAY, fill_date, rebalance_dates, signal_date
from .score import score_at
from .store import prices_wide
from .universe import Asset, Config, Strategy

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
    # Number of (entrant, exit) swap proposals the score-band filter
    # rejected at this rebalance. The rejected entrants are rolled back to
    # their paired exits — the unchanged held assets keep their rule-driven
    # weights (so within-set score-prop reweighting still applies). For
    # top-1 strategies, swaps_rejected > 0 implies turnover == 0 (the only
    # swap was undone). For top-N score-prop strategies, swaps_rejected > 0
    # is consistent with turnover > 0 (the within-set reweight produced
    # non-zero L1).
    swaps_rejected: int = 0


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
    safe_id = config.safe_asset_id  # None if no asset has synth_proxy=estr
    # Residual-holder rule: rounding shortfalls and the "no scores at all
    # at this rebalance" early-history fallback go to the safe asset *if
    # it is listed* in the strategy's universe (so the residual earns
    # €STR yield), otherwise to CASH_KEY (a 0%-return placeholder).
    # (Under the rank-only methodology there is no "no candidate passed
    # the filter" path; even all-negative scores get allocated.)
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
        return _run_buy_and_hold(wide, strategy, asset_ids, backtest_start=start)

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
    if safe_id is not None:
        half_spread_by_id[safe_id] = 0.0
    half_spread_by_id[CASH_KEY] = 0.0
    scoring = strategy.effective_scoring(config.shared.scoring)
    date_set = set(dates)

    rebalances: list[Rebalance] = []
    fills: list[tuple[date, dict[str, float], float]] = []
    prev_weights: dict[str, float] = {residual_holder: 1.0}
    n_fill_skips = 0
    n_signal_skips = 0
    threshold_frac = strategy.effective_threshold_pct(config.shared.scoring) / 100.0
    had_real_rebalance = False
    last_real_scores: dict[str, float] = {}

    asset_by_id = {a.id: a for a in config.assets}
    for r_day in rebalance_dates(strategy, dates[0], dates[-1]):
        s_day = signal_date(r_day)
        f_day = fill_date(r_day)
        if f_day not in date_set:
            n_fill_skips += 1
            continue
        scores = score_at(prices_long, asset_ids, s_day, scoring)
        if strategy.selection_rule == "top_1_per_region":
            scores = _filter_top_1_per_region(scores, asset_by_id)
        if not scores:
            n_signal_skips += 1
            continue
        new_w_proposed = allocate(
            scores=scores,
            top_n=strategy.top_n,
            alloc=config.shared.allocation,
            rule_override=strategy.allocation_rule,
            residual_holder=residual_holder,
            regional_weights=strategy.regional_weights,
            asset_by_id=asset_by_id,
        )
        new_w, n_rejected = _apply_band(
            had_real_rebalance=had_real_rebalance,
            threshold_frac=threshold_frac,
            prev_weights=prev_weights,
            new_w=new_w_proposed,
            scores=scores,
            last_real_scores=last_real_scores,
            residual_holder=residual_holder,
            r_day=r_day,
            strategy_name=strategy.name,
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
                swaps_rejected=n_rejected,
            )
        )
        # A rebalance with zero turnover doesn't enter the fills list (no
        # transition for the compounding kernel). Per-swap rejection can
        # still produce non-zero turnover (within-set reweight in score-
        # prop), in which case it's a real fill. `had_real_rebalance` flips
        # on the first iteration unconditionally so the gate is active
        # from the second onwards.
        if turnover > 0:
            fills.append((f_day, new_w, cost))
        prev_weights = new_w
        last_real_scores = scores
        had_real_rebalance = True

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


def _filter_top_1_per_region(
    scores: dict[str, float],
    asset_by_id: dict[str, Asset],
) -> dict[str, float]:
    """Keep only the highest-scoring asset within each {us, europe, asia}
    bucket. Returns a scores dict with at most 3 entries.

    Region is inferred from `discover.dashboard_bucket(asset.category)`.
    Assets that don't map to one of `REGIONAL_BUCKETS` (cash, world,
    unmapped) are dropped — this rule is for region-rotation strategies
    only. Ties are broken by asset_id (deterministic, but ties on a
    floating-point score are vanishingly rare in practice).
    """
    grouped = assets_by_region(scores, asset_by_id)
    out: dict[str, float] = {}
    for bucket_assets in grouped.values():
        if not bucket_assets:
            continue
        best_id = max(bucket_assets, key=lambda aid: scores[aid])
        out[best_id] = scores[best_id]
    return out


def _turnover(prev: dict[str, float], new: dict[str, float]) -> float:
    """L1 distance between two weight dicts. Safe-asset rebalancing also counts."""
    keys = set(prev) | set(new)
    return sum(abs(new.get(k, 0.0) - prev.get(k, 0.0)) for k in keys)


def _apply_band(
    *,
    had_real_rebalance: bool,
    threshold_frac: float,
    prev_weights: dict[str, float],
    new_w: dict[str, float],
    scores: dict[str, float],
    last_real_scores: dict[str, float],
    residual_holder: str,
    r_day: date,
    strategy_name: str,
) -> tuple[dict[str, float], int]:
    """Per-swap score-band gate.

    Identifies (entrant, exit) pairs in the proposed rebalance, pairs them
    by score (weakest entrant ↔ strongest exit), and rejects any pair whose
    score gap is below `threshold_frac`. Rejected entrants have their
    allocate-time weight transferred back to the exit they would have
    displaced — the unchanged held assets keep their rule-driven weights,
    so score-proportional within-set reweighting still happens even when
    some swaps are rejected.

    Returns `(adjusted_weights, n_rejected)`. The first transition out of
    the initial cash state is never gated.

    Pairing strategy — entrants asc by score, exits desc by score, then
    zip — pairs the marginal entrant (most likely noise) with the
    strongest exit (highest "should I keep this?" pull). This is the
    optimal assignment for "maximize swaps whose gap clears the band".
    """
    if threshold_frac <= 0.0 or not had_real_rebalance:
        return new_w, 0
    held_prev = {k for k, v in prev_weights.items() if v > 0}
    held_new = {k for k, v in new_w.items() if v > 0}
    entrants = held_new - held_prev
    exits = held_prev - held_new
    if not entrants or not exits:
        return new_w, 0

    def _score_for(aid: str) -> float:
        if aid in scores:
            return scores[aid]
        if aid in last_real_scores:
            return last_real_scores[aid]
        if aid == residual_holder or aid == CASH_KEY:
            return 0.0
        return float("-inf")

    sorted_entrants = sorted(entrants, key=_score_for)  # asc — weakest first
    sorted_exits = sorted(exits, key=_score_for, reverse=True)  # desc — strongest first

    adjusted = dict(new_w)
    n_rejected = 0
    # zip stops at the shorter of the two — unpaired entrants (no exit to
    # displace) and unpaired exits (just dropped, no displacement) are
    # outside the gate's purview.
    for entrant, exit_ in zip(sorted_entrants, sorted_exits, strict=False):
        gap = _score_for(entrant) - _score_for(exit_)
        if gap < threshold_frac:
            w = adjusted.pop(entrant, 0.0)
            adjusted[exit_] = adjusted.get(exit_, 0.0) + w
            n_rejected += 1
            log.info(
                "%s swap-reject @ %s: %s(%.4f) ↛ %s(%.4f) gap=%.4f < %.4f",
                strategy_name,
                r_day,
                exit_,
                _score_for(exit_),
                entrant,
                _score_for(entrant),
                gap,
                threshold_frac,
            )
    return adjusted, n_rejected


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
    backtest_start: date | None = None,
) -> BacktestResult:
    """Buy-and-hold benchmark. Default = equal-weight across `asset_ids`;
    when `strategy.static_weights` is set (e.g. 60/40), use those instead.
    Static weights may reference the safe asset id. No rebalances, no
    transaction costs.

    Equity curve start alignment: when `backtest_start` is provided the
    curve is aligned to the first Sunday of the backtest range — the
    same "day 1" rotation strategies use for their first rebalance.
    This makes B&H benchmarks directly comparable to rotation strategies
    in cross-strategy charts. If some held asset has no data on the
    first Sunday, the curve falls back to the first day all assets are
    available (latest-listed-sleeve's first day) — alignment is
    achievable only when the data permits.
    """
    if strategy.static_weights is not None:
        weights = dict(strategy.static_weights)
        asset_cols = list(weights)
        missing = [a for a in asset_cols if a not in wide.columns]
        if missing:
            raise ValueError(
                f"buy_and_hold {strategy.name!r}: static_weights references "
                f"asset(s) {missing} that have no fetched price data. "
                f"YAML validation enforces static_weights keys == assets:, so "
                f"this means the fetch step is incomplete — re-run fetch or "
                f"remove the asset from the strategy."
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

    # Cross-strategy start alignment: slide the equity curve forward to
    # the first Sunday of the backtest range when possible (no-op when
    # the latest-listed sleeve's first available day is already past
    # that Sunday). Same first-Sunday-of-backtest day rotations fire on.
    if backtest_start is not None:
        first_sunday = backtest_start + timedelta(days=(_SUNDAY - backtest_start.weekday()) % 7)
        wide_aligned = wide.filter(pl.col("date") >= first_sunday)
        if not wide_aligned.is_empty():
            wide = wide_aligned

    dates = wide.get_column("date").to_list()
    equity_df = _compound(wide, asset_cols, init_weights=weights, fills=[])

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
                "swaps_rejected": r.swaps_rejected,
            }
            for r in rebalances
        ],
        indent=2,
        default=str,
    )


def rebalances_from_json(text: str) -> list[Rebalance]:
    """Deserialise the artifact written by `rebalances_to_json`.

    Required keys raise KeyError if missing (corrupted artifact).
    Backward-compat for two earlier schemas:
    - Pre-band artifacts (no rejection field) → swaps_rejected = 0.
    - All-or-nothing band schema (`band_skipped: bool`) → swaps_rejected
      = 1 if True else 0. The exact rejection count was never recorded
      in that schema; 1 is the closest faithful approximation (≥ 1).
    """
    raw = json.loads(text)

    def _swaps_rejected(r: dict[str, object]) -> int:
        if "swaps_rejected" in r:
            return int(r["swaps_rejected"])  # type: ignore[arg-type]
        if r.get("band_skipped"):
            return 1
        return 0

    return [
        Rebalance(
            rebalance_date=date.fromisoformat(r["rebalance_date"]),
            signal_date=date.fromisoformat(r["signal_date"]),
            fill_date=date.fromisoformat(r["fill_date"]),
            scores=r["scores"],
            weights=r["weights"],
            turnover=float(r["turnover"]),
            cost=float(r["cost"]),
            swaps_rejected=_swaps_rejected(r),
        )
        for r in raw
    ]
