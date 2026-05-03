"""End-to-end smoke tests for the backtest engine on synthetic data."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from pea_momentum.backtest import run
from pea_momentum.universe import (
    Allocation,
    Asset,
    Config,
    Costs,
    Scoring,
    Shared,
    Strategy,
)


def _config() -> Config:
    return Config(
        shared=Shared(
            scoring=Scoring(lookbacks_days=(21, 63, 126), aggregation="mean"),
            allocation=Allocation(
                rule="score_proportional",
                granularity_pct=10,
                rounding="largest_remainder",
            ),
            costs=Costs(per_trade_pct=0.10),
        ),
        assets=(
            Asset(id="up", isin="x", yahoo="x"),
            Asset(id="dn", isin="x", yahoo="x"),
            Asset(id="safe", isin="x", yahoo="", synth_proxy="estr"),
        ),
        strategies=(),
    )


def _strategy(rebalance: str = "weekly_sunday") -> Strategy:
    return Strategy(
        name="t",
        asset_ids=("up", "dn"),
        rebalance=rebalance,
        top_n=1,
        reference_date=None,
    )


def _synthetic_prices(start: date, days: int) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for i in range(days):
        d = start + timedelta(days=i)
        rows.append({"date": d, "asset_id": "up", "close": 100.0 * (1.0005**i)})
        rows.append({"date": d, "asset_id": "dn", "close": 100.0 * (0.9995**i)})
        rows.append({"date": d, "asset_id": "safe", "close": 100.0 * (1.0001**i)})
    return pl.DataFrame(rows).cast({"date": pl.Date, "asset_id": pl.Utf8, "close": pl.Float64})


def test_backtest_runs_end_to_end() -> None:
    prices = _synthetic_prices(date(2023, 1, 1), days=400)
    result = run(prices, _strategy(), _config())
    assert not result.equity.is_empty()
    assert result.equity.get_column("equity")[0] == 1.0
    assert len(result.rebalances) > 0


def test_backtest_picks_uptrending_asset() -> None:
    prices = _synthetic_prices(date(2023, 1, 1), days=400)
    result = run(prices, _strategy(), _config())
    final_weights = result.rebalances[-1].weights
    assert final_weights.get("up", 0.0) >= 0.9
    assert final_weights.get("dn", 0.0) == 0.0


def test_backtest_equity_grows_with_uptrend() -> None:
    prices = _synthetic_prices(date(2023, 1, 1), days=400)
    result = run(prices, _strategy(), _config())
    final_equity = result.equity.get_column("equity")[-1]
    assert final_equity > 1.0


def test_monthly_has_fewer_rebalances_than_weekly() -> None:
    prices = _synthetic_prices(date(2023, 1, 1), days=400)
    weekly = run(prices, _strategy("weekly_sunday"), _config())
    monthly = run(prices, _strategy("monthly_first_sunday"), _config())
    assert len(monthly.rebalances) < len(weekly.rebalances)


def test_no_rebalance_with_too_short_history() -> None:
    prices = _synthetic_prices(date(2023, 1, 1), days=50)
    result = run(prices, _strategy(), _config())
    assert result.rebalances == []


def _bh_strategy() -> Strategy:
    return Strategy(
        name="bh",
        asset_ids=("up",),
        rebalance="monthly_first_sunday",
        top_n=1,
        reference_date=None,
        mode="buy_and_hold",
    )


def test_buy_and_hold_holds_constant_weight() -> None:
    prices = _synthetic_prices(date(2023, 1, 1), days=400)
    result = run(prices, _bh_strategy(), _config())
    assert len(result.rebalances) == 1
    assert result.rebalances[0].weights == {"up": 1.0}
    assert result.rebalances[0].turnover == 0.0
    assert result.rebalances[0].cost == 0.0


def test_buy_and_hold_zero_cost() -> None:
    prices = _synthetic_prices(date(2023, 1, 1), days=400)
    result = run(prices, _bh_strategy(), _config())
    total_cost = sum(r.cost for r in result.rebalances)
    assert total_cost == 0.0


def test_buy_and_hold_tracks_underlying_uptrend() -> None:
    prices = _synthetic_prices(date(2023, 1, 1), days=400)
    result = run(prices, _bh_strategy(), _config())
    # `up` grows at 0.05% per day; over ~400 days that's ~22% gain.
    final_eq = result.equity.get_column("equity")[-1]
    assert 1.15 < final_eq < 1.30


def _static_60_40_strategy() -> Strategy:
    return Strategy(
        name="static_60_40",
        asset_ids=("up",),
        rebalance="monthly_first_sunday",
        top_n=1,
        reference_date=None,
        mode="buy_and_hold",
        static_weights=(("up", 0.6), ("safe", 0.4)),
    )


def test_static_60_40_uses_explicit_weights() -> None:
    prices = _synthetic_prices(date(2023, 1, 1), days=400)
    result = run(prices, _static_60_40_strategy(), _config())
    assert result.rebalances[0].weights == {"up": 0.6, "safe": 0.4}
    # Final equity must lie between safe-only (0.04% * 400 ~ 4%) and up-only
    # (0.05% * 400 ~ 22%) - this is the 60/40 mix sanity check.
    final_eq = result.equity.get_column("equity")[-1]
    assert 1.04 < final_eq < 1.22


def test_per_asset_spread_increases_total_cost() -> None:
    """An asset with high est_spread_bps charges more per turnover than one
    without. Configure two strategies with the same underlying signals but
    one asset wide-spread, the other tight-spread; the wide-spread strategy
    should have a strictly larger total cost over the run."""
    base_cfg = _config()
    safe_asset = next(a for a in base_cfg.assets if a.synth_proxy == "estr")
    cfg_tight = Config(
        shared=base_cfg.shared,
        assets=(
            Asset(id="up", isin="x", yahoo="x", est_spread_bps=0.0),
            Asset(id="dn", isin="x", yahoo="x", est_spread_bps=0.0),
            safe_asset,
        ),
        strategies=(),
    )
    cfg_wide = Config(
        shared=base_cfg.shared,
        assets=(
            Asset(id="up", isin="x", yahoo="x", est_spread_bps=100.0),
            Asset(id="dn", isin="x", yahoo="x", est_spread_bps=100.0),
            safe_asset,
        ),
        strategies=(),
    )
    prices = _synthetic_prices(date(2023, 1, 1), days=400)
    cost_tight = sum(r.cost for r in run(prices, _strategy(), cfg_tight).rebalances)
    cost_wide = sum(r.cost for r in run(prices, _strategy(), cfg_wide).rebalances)
    assert cost_wide > cost_tight
