"""End-to-end smoke tests for the backtest engine on synthetic data."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

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
        asset_ids=("up", "safe"),
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


# ── Per-region selection rule ───────────────────────────────────────────


def test_filter_top_1_per_region_keeps_best_in_each_bucket() -> None:
    """`_filter_top_1_per_region` groups assets by `dashboard_bucket(category)`
    and keeps the highest-scoring asset per region. Cross-region selection
    is independent — picking the best in `us` doesn't affect `europe` or
    `asia`."""
    from pea_momentum.backtest import _filter_top_1_per_region

    asset_by_id = {
        "us_a": Asset(id="us_a", isin="x", yahoo="x", category="USA"),
        "us_b": Asset(id="us_b", isin="x", yahoo="x", category="USA-Tech"),
        "eu_a": Asset(id="eu_a", isin="x", yahoo="x", category="Eurozone"),
        "eu_b": Asset(id="eu_b", isin="x", yahoo="x", category="France"),
        "as_a": Asset(id="as_a", isin="x", yahoo="x", category="Japan"),
        "world_x": Asset(id="world_x", isin="x", yahoo="x", category="World"),
        "cash_a": Asset(id="cash_a", isin="x", yahoo="x", category="Cash-Eurozone"),
    }
    scores = {
        "us_a": 0.10,
        "us_b": 0.20,  # best US
        "eu_a": 0.05,  # best Europe (eu_b is lower)
        "eu_b": 0.02,
        "as_a": -0.05,  # only Asia → wins regardless of sign
        "world_x": 0.50,  # world bucket → dropped
        "cash_a": 0.01,  # cash bucket → dropped
    }
    result = _filter_top_1_per_region(scores, asset_by_id)
    assert set(result.keys()) == {"us_b", "eu_a", "as_a"}
    assert result["us_b"] == 0.20
    assert result["eu_a"] == 0.05
    assert result["as_a"] == -0.05


def test_filter_top_1_per_region_handles_empty_buckets() -> None:
    """If a region has no asset in scores, it's simply absent from the
    result. The downstream allocate then sees fewer than 3 entries."""
    from pea_momentum.backtest import _filter_top_1_per_region

    asset_by_id = {
        "us_a": Asset(id="us_a", isin="x", yahoo="x", category="USA"),
        "as_a": Asset(id="as_a", isin="x", yahoo="x", category="Japan"),
    }
    scores = {"us_a": 0.10, "as_a": 0.05}
    result = _filter_top_1_per_region(scores, asset_by_id)
    assert set(result.keys()) == {"us_a", "as_a"}


def test_filter_top_1_per_region_drops_world_and_cash_buckets() -> None:
    """Only assets in {us, europe, asia} buckets compete; world and cash
    are dropped. Region inference uses `discover.dashboard_bucket` —
    note that uncategorised / catchall categories (e.g. `Thematic-*`,
    `Sector-*` for European exposures) fall into `europe` per
    `dashboard_bucket`'s default branch and so DO compete in the europe
    bucket. If a strategy wants to exclude those, it should leave them
    out of `assets:`."""
    from pea_momentum.backtest import _filter_top_1_per_region

    asset_by_id = {
        "us_a": Asset(id="us_a", isin="x", yahoo="x", category="USA"),
        "world_x": Asset(id="world_x", isin="x", yahoo="x", category="World"),
        "cash_a": Asset(id="cash_a", isin="x", yahoo="x", category="Cash-Eurozone"),
    }
    scores = {"us_a": 0.10, "world_x": 0.50, "cash_a": 0.30}
    result = _filter_top_1_per_region(scores, asset_by_id)
    assert set(result.keys()) == {"us_a"}


# ── Regional fixed-weight rule (end-to-end via run() — unit tests in test_allocate) ──


def _regional_static_strategy() -> Strategy:
    """Top-1-per-region selection with US=60% / EU=10% / Asia=30% fixed
    split. Uses the synthetic `up` (USA) / `dn` (USA) / `safe` assets
    of `_config()` plus extra Europe/Asia assets the test injects."""
    return Strategy(
        name="regional_static",
        asset_ids=("up", "eu", "as"),
        rebalance="weekly_sunday",
        top_n=3,
        lookbacks_days=(63,),
        reference_date=None,
        selection_rule="top_1_per_region",
        regional_weights=(("us", 0.6), ("europe", 0.1), ("asia", 0.3)),
    )


def test_regional_weights_produce_configured_split() -> None:
    """End-to-end: with one asset per region (so the top-1-per-region
    filter keeps all three) and a 60/10/30 regional split, the final
    weights match the configuration after granularity rounding."""
    cfg = Config(
        shared=_config().shared,
        assets=(
            Asset(id="up", isin="x", yahoo="x", category="USA"),
            Asset(id="eu", isin="x", yahoo="x", category="Eurozone"),
            Asset(id="as", isin="x", yahoo="x", category="Japan"),
        ),
        strategies=(),
    )
    rows: list[dict[str, object]] = []
    for i in range(400):
        d = date(2023, 1, 1) + timedelta(days=i)
        rows.append({"date": d, "asset_id": "up", "close": 100.0 * (1.0005**i)})
        rows.append({"date": d, "asset_id": "eu", "close": 100.0 * (1.0003**i)})
        rows.append({"date": d, "asset_id": "as", "close": 100.0 * (1.0004**i)})
    prices = pl.DataFrame(rows).cast({"date": pl.Date, "asset_id": pl.Utf8, "close": pl.Float64})
    result = run(prices, _regional_static_strategy(), cfg)
    final = result.rebalances[-1].weights
    assert final.get("up", 0.0) == 0.6
    assert final.get("eu", 0.0) == 0.1
    assert final.get("as", 0.0) == 0.3


def test_regional_weights_renormalise_when_region_missing() -> None:
    """If only US and Asia have eligible assets at a rebalance (Europe
    absent from the universe), the 0.6/0.3 targets renormalise to
    ~0.667/0.333. After 10% granularity rounding that becomes 70/30."""
    cfg = Config(
        shared=_config().shared,
        assets=(
            Asset(id="up", isin="x", yahoo="x", category="USA"),
            Asset(id="as", isin="x", yahoo="x", category="Japan"),
        ),
        strategies=(),
    )
    rows: list[dict[str, object]] = []
    for i in range(400):
        d = date(2023, 1, 1) + timedelta(days=i)
        rows.append({"date": d, "asset_id": "up", "close": 100.0 * (1.0005**i)})
        rows.append({"date": d, "asset_id": "as", "close": 100.0 * (1.0004**i)})
    prices = pl.DataFrame(rows).cast({"date": pl.Date, "asset_id": pl.Utf8, "close": pl.Float64})
    strategy = Strategy(
        name="regional_static_2",
        asset_ids=("up", "as"),
        rebalance="weekly_sunday",
        top_n=3,
        lookbacks_days=(63,),
        reference_date=None,
        selection_rule="top_1_per_region",
        regional_weights=(("us", 0.6), ("europe", 0.1), ("asia", 0.3)),
    )
    result = run(prices, strategy, cfg)
    final = result.rebalances[-1].weights
    assert final.get("up", 0.0) == 0.7
    assert final.get("as", 0.0) == 0.3


# ── Score-band hysteresis filter (whipsaw suppression) ──────────────────


def _flipping_prices(start: date, days: int) -> pl.DataFrame:
    """Two near-identical assets that swap trivially around day ~150 and
    then swap back around ~250 — a synthetic whipsaw scenario where the
    ranks flip but the score gap is small (sub-1%). The band filter
    should suppress the whipsaws when the threshold exceeds the gap.
    """
    rows: list[dict[str, object]] = []
    for i in range(days):
        d = start + timedelta(days=i)
        # Two slow-grinding equities with cross-overs at days ~150 / ~250.
        # Both grow ~equally (~5bp/day) with a tiny phase shift so the
        # rank flips but the magnitude difference stays below ~1%.
        a = 100.0 * (1.0006**i) * (1 + 0.005 * (1 if 100 <= i <= 200 else 0))
        b = 100.0 * (1.0006**i) * (1 + 0.005 * (1 if 200 < i <= 300 else 0))
        rows.append({"date": d, "asset_id": "ax", "close": a})
        rows.append({"date": d, "asset_id": "bx", "close": b})
        rows.append({"date": d, "asset_id": "safe", "close": 100.0 * (1.0001**i)})
    return pl.DataFrame(rows).cast({"date": pl.Date, "asset_id": pl.Utf8, "close": pl.Float64})


def _band_config() -> Config:
    return Config(
        shared=Shared(
            scoring=Scoring(lookbacks_days=(21, 63), aggregation="mean"),
            allocation=Allocation(
                rule="equal_weight",
                granularity_pct=10,
                rounding="largest_remainder",
            ),
            costs=Costs(per_trade_pct=0.10),
        ),
        assets=(
            Asset(id="ax", isin="x", yahoo="x"),
            Asset(id="bx", isin="x", yahoo="x"),
            Asset(id="safe", isin="x", yahoo="", synth_proxy="estr"),
        ),
        strategies=(),
    )


def _band_strategy(threshold_pct: float | None) -> Strategy:
    return Strategy(
        name="band",
        asset_ids=("ax", "bx"),
        rebalance="weekly_sunday",
        top_n=1,
        reference_date=None,
        momentum_delta_threshold_pct=threshold_pct,
    )


def test_band_filter_disabled_matches_baseline() -> None:
    """threshold=0 must be byte-identical to the prior behaviour: no
    swaps_rejected counts ever non-zero."""
    prices = _flipping_prices(date(2023, 1, 1), days=400)
    result = run(prices, _band_strategy(threshold_pct=0.0), _band_config())
    assert all(r.swaps_rejected == 0 for r in result.rebalances)


def test_band_filter_suppresses_tight_swaps() -> None:
    """A high threshold (10%) on noise-grade rotations must trigger
    swap rejections and reduce the count of actual reallocations."""
    prices = _flipping_prices(date(2023, 1, 1), days=400)
    off = run(prices, _band_strategy(threshold_pct=0.0), _band_config())
    on = run(prices, _band_strategy(threshold_pct=10.0), _band_config())
    n_swaps_off = sum(1 for r in off.rebalances if r.turnover > 0)
    n_swaps_on = sum(1 for r in on.rebalances if r.turnover > 0)
    assert n_swaps_on < n_swaps_off
    assert any(r.swaps_rejected > 0 for r in on.rebalances)
    # Top-1 strategy: when the only swap is rejected, the rolled-back
    # weights equal prev, so turnover and cost are zero.
    for r in on.rebalances:
        if r.swaps_rejected > 0:
            assert r.turnover == 0.0
            assert r.cost == 0.0


def test_band_filter_does_not_block_first_rebalance() -> None:
    """Initial state is `{residual_holder: 1.0}` — the first transition
    out of cash is never an oscillation and must not see any swap
    rejection even with a very large threshold."""
    prices = _flipping_prices(date(2023, 1, 1), days=400)
    result = run(prices, _band_strategy(threshold_pct=1000.0), _band_config())
    assert result.rebalances, "expected at least one rebalance"
    assert result.rebalances[0].swaps_rejected == 0


def test_band_filter_per_strategy_override_zero_disables() -> None:
    """An explicit 0 on the strategy must disable the band even if the
    shared scoring has a non-zero default."""
    cfg = _band_config()
    shared_with_default = Shared(
        scoring=Scoring(
            lookbacks_days=(21, 63),
            aggregation="mean",
            momentum_delta_threshold_pct=10.0,
        ),
        allocation=cfg.shared.allocation,
        costs=cfg.shared.costs,
    )
    cfg = Config(shared=shared_with_default, assets=cfg.assets, strategies=())
    prices = _flipping_prices(date(2023, 1, 1), days=400)
    result = run(prices, _band_strategy(threshold_pct=0.0), cfg)
    assert all(r.swaps_rejected == 0 for r in result.rebalances)


def test_rebalances_json_round_trip_preserves_swaps_rejected() -> None:
    from pea_momentum.backtest import rebalances_from_json, rebalances_to_json

    prices = _flipping_prices(date(2023, 1, 1), days=400)
    on = run(prices, _band_strategy(threshold_pct=10.0), _band_config())
    text = rebalances_to_json(on.rebalances)
    parsed = rebalances_from_json(text)
    assert len(parsed) == len(on.rebalances)
    for orig, got in zip(on.rebalances, parsed, strict=True):
        assert orig.swaps_rejected == got.swaps_rejected
        assert orig.turnover == got.turnover


def test_rebalances_from_json_backward_compat_band_skipped_bool() -> None:
    """Artifacts written under the brief band_skipped:bool schema must
    deserialise: True → swaps_rejected=1, False → 0."""
    import json

    from pea_momentum.backtest import rebalances_from_json

    text = json.dumps(
        [
            {
                "rebalance_date": "2023-01-01",
                "signal_date": "2022-12-30",
                "fill_date": "2023-01-02",
                "scores": {"a": 0.05},
                "weights": {"a": 1.0},
                "turnover": 0.0,
                "cost": 0.0,
                "band_skipped": True,
            },
            {
                "rebalance_date": "2023-02-01",
                "signal_date": "2023-01-30",
                "fill_date": "2023-02-02",
                "scores": {"a": 0.06},
                "weights": {"a": 1.0},
                "turnover": 0.0,
                "cost": 0.0,
                "band_skipped": False,
            },
        ]
    )
    parsed = rebalances_from_json(text)
    assert parsed[0].swaps_rejected == 1
    assert parsed[1].swaps_rejected == 0


# ── Per-swap surgical gate ──────────────────────────────────────────────


def test_per_swap_partial_acceptance_top3_score_prop() -> None:
    """Top-3 score-prop with 2 simultaneous swaps where only one is below
    the band: the strong swap goes through, the weak swap is rolled back.

    Universe: A B C D E. Prev held = {A B C}.
    New scores craft: D enters cleanly (beats C by margin), E is marginal
    (gap < threshold against B).
    Expected: held set becomes {A D B} (E rejected, B retained); D's
    allocate-time weight goes to D, E's allocate-time weight goes to B.
    """
    cfg = Config(
        shared=Shared(
            scoring=Scoring(lookbacks_days=(21,), aggregation="mean"),
            allocation=Allocation(
                rule="score_proportional",
                granularity_pct=10,
                rounding="largest_remainder",
            ),
            costs=Costs(per_trade_pct=0.0),
        ),
        assets=tuple(Asset(id=x, isin="x", yahoo="x") for x in "abcde"),
        strategies=(),
    )
    rows: list[dict[str, object]] = []
    days = 400
    for i in range(days):
        d = date(2023, 1, 1) + timedelta(days=i)
        # Phase 1 (i < 220): A best, then B, then C; D and E weak.
        # Phase 2 (i ≥ 220): D climbs strongly past C; E creeps to just
        # under B (gap < band threshold). A stays leader.
        if i < 220:
            mults = {"a": 1.0010, "b": 1.0008, "c": 1.0005, "d": 1.0001, "e": 1.0001}
        else:
            mults = {"a": 1.0010, "b": 1.0008, "c": 1.0005, "d": 1.0015, "e": 1.00079}
        for k, m in mults.items():
            rows.append({"date": d, "asset_id": k, "close": 100.0 * (m**i)})
    prices = pl.DataFrame(rows).cast({"date": pl.Date, "asset_id": pl.Utf8, "close": pl.Float64})
    s = Strategy(
        name="surg",
        asset_ids=("a", "b", "c", "d", "e"),
        rebalance="weekly_sunday",
        top_n=3,
        reference_date=None,
        momentum_delta_threshold_pct=2.0,
    )
    result = run(prices, s, cfg)
    # Find a rebalance where multiple membership changes are proposed.
    # Iterate post-warmup (after a few rebalances) and find one with
    # swaps_rejected > 0.
    rejs = [r for r in result.rebalances if r.swaps_rejected > 0]
    assert rejs, "expected at least one rebalance with a rejected swap"
    # On rejected rebalances, the rejected entrants should be absent from
    # the final weights and the saved exit should be present.
    for r in rejs:
        assert sum(r.weights.values()) == pytest.approx(1.0, rel=1e-6)


def test_per_swap_within_set_reweight_passes_through() -> None:
    """Score-proportional top-3 where the held set doesn't change but
    scores diverge: turnover should be > 0 (within-set reweight) and
    swaps_rejected == 0 (no membership swap to evaluate)."""
    cfg = Config(
        shared=Shared(
            scoring=Scoring(lookbacks_days=(21,), aggregation="mean"),
            allocation=Allocation(
                rule="score_proportional",
                granularity_pct=1,  # fine granularity so reweights register
                rounding="largest_remainder",
            ),
            costs=Costs(per_trade_pct=0.0),
        ),
        assets=tuple(Asset(id=x, isin="x", yahoo="x") for x in "abc"),
        strategies=(),
    )
    # Time-varying growth so the score *ratios* drift over the backtest:
    # A and B exchange leadership every ~80 days; C grinds steadily. The
    # held set stays {a, b, c} (top-3 of 3) but the score-prop weights
    # shift across rebalance dates.
    import math

    rows: list[dict[str, object]] = []
    for i in range(400):
        d = date(2023, 1, 1) + timedelta(days=i)
        a_rate = 1.0008 + 0.001 * math.sin(i / 40)
        b_rate = 1.0008 - 0.001 * math.sin(i / 40)
        c_rate = 1.0006
        rows.append({"date": d, "asset_id": "a", "close": 100.0 * (a_rate**i)})
        rows.append({"date": d, "asset_id": "b", "close": 100.0 * (b_rate**i)})
        rows.append({"date": d, "asset_id": "c", "close": 100.0 * (c_rate**i)})
    prices = pl.DataFrame(rows).cast({"date": pl.Date, "asset_id": pl.Utf8, "close": pl.Float64})
    s = Strategy(
        name="reweight",
        asset_ids=("a", "b", "c"),
        rebalance="weekly_sunday",
        top_n=3,
        reference_date=None,
        momentum_delta_threshold_pct=10.0,  # high threshold; held set is fixed anyway
    )
    result = run(prices, s, cfg)
    # Top-3 of 3 assets => held set is always {a, b, c} → no entrants /
    # exits ever → swaps_rejected always 0; within-set reweight produces
    # at least one turnover > 0 rebalance.
    assert all(r.swaps_rejected == 0 for r in result.rebalances)
    assert any(r.turnover > 0 for r in result.rebalances[1:]), \
        "expected at least one within-set reweight after the first rebalance"
