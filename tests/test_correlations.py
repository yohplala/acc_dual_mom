"""Tests for correlation matrix + group identification on synthetic data."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from pea_momentum.correlations import (
    CorrelationMatrix,
    GroupRepresentative,
    best_in_group,
    compute_correlation_matrix,
    find_groups,
)
from pea_momentum.diagnostics import diagnose_strategies
from pea_momentum.discover import DiscoveryEntry
from pea_momentum.universe import (
    Allocation,
    Asset,
    Config,
    Costs,
    Scoring,
    Shared,
    Strategy,
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

    def test_complete_link_breaks_bridge_chain(self) -> None:
        """Complete-link clustering: a-b 0.90 and b-c 0.90 alone don't justify
        unioning a with c when their direct correlation (0.50) is below
        threshold. a joins (a,b), then c sees a-c 0.50 < threshold and opens
        its own group."""
        cm = CorrelationMatrix(
            asset_ids=["a", "b", "c"],
            matrix=np.array([[1.0, 0.90, 0.50], [0.90, 1.0, 0.90], [0.50, 0.90, 1.0]]),
            window_days=252,
        )
        groups = find_groups(cm, threshold=0.85)
        # 2 groups: (a, b) and (c,)
        assert len(groups) == 2
        assert sorted(groups[0]) == ["a", "b"]
        assert groups[1] == ["c"]

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
    def test_picks_lowest_ter(self) -> None:
        rep = best_in_group(["a", "b", "c"], ter_pct_by_id={"a": 0.20, "b": 0.30, "c": 0.10})
        assert rep.representative == "c"

    def test_missing_ter_loses(self) -> None:
        # `b` has no TER → infinite cost → never wins.
        rep = best_in_group(["a", "b"], ter_pct_by_id={"a": 0.10})
        assert rep.representative == "a"

    def test_singleton_group(self) -> None:
        rep = best_in_group(["a"], ter_pct_by_id={"a": 0.10})
        assert rep.representative == "a"
        assert rep.group == ["a"]


class TestDiagnoseStrategies:
    """Cross-strategy diagnostics: 'remove' (redundant pair) + 'replace'
    (suboptimal pick within a group)."""

    def _make_setup(self) -> tuple[Config, list[DiscoveryEntry], list[GroupRepresentative]]:
        """A 3-strategy / 3-group setup for testing both diagnostic types."""

        # 4 active assets + safe sleeve, ISINs that overlap with the
        # discovery_entries fixture below.
        assets = (
            Asset(id="us", isin="ISIN_US", yahoo="x"),
            Asset(id="us_alt", isin="ISIN_US_ALT", yahoo="x"),
            Asset(id="eu", isin="ISIN_EU", yahoo="x"),
            Asset(id="jp", isin="ISIN_JP", yahoo="x"),
            Asset(id="safe", isin="ISIN_SAFE", yahoo="", synth_proxy="estr"),
        )
        cfg = Config(
            shared=Shared(
                scoring=Scoring(lookbacks_days=(126,), aggregation="mean"),
                allocation=Allocation(
                    rule="score_proportional",
                    granularity_pct=10,
                    rounding="largest_remainder",
                ),
                costs=Costs(per_trade_pct=0.10),
            ),
            assets=assets,
            strategies=(
                # Strategy A: uses both us AND us_alt (redundant pair)
                Strategy(
                    name="strat_redundant",
                    asset_ids=("us", "us_alt", "eu"),
                    rebalance="monthly_first_sunday",
                    top_n=2,
                ),
                # Strategy B: uses us_alt (suboptimal — us is the rep)
                Strategy(
                    name="strat_suboptimal",
                    asset_ids=("us_alt", "eu"),
                    rebalance="monthly_first_sunday",
                    top_n=2,
                ),
                # Strategy C: uses us (the rep) — clean, no diagnostics
                Strategy(
                    name="strat_clean",
                    asset_ids=("us", "eu", "jp"),
                    rebalance="monthly_first_sunday",
                    top_n=2,
                ),
            ),
        )

        # Discovery universe — same ISINs, different ids
        entries = [
            DiscoveryEntry(
                id="d_us",
                name="us",
                isin="ISIN_US",
                currency="EUR",
                ter_pct=0.12,
                category="USA",
                yahoo="x",
            ),
            DiscoveryEntry(
                id="d_us_alt",
                name="us_alt",
                isin="ISIN_US_ALT",
                currency="EUR",
                ter_pct=0.30,
                category="USA",
                yahoo="x",
            ),
            DiscoveryEntry(
                id="d_eu",
                name="eu",
                isin="ISIN_EU",
                currency="EUR",
                ter_pct=0.15,
                category="Europe",
                yahoo="x",
            ),
            DiscoveryEntry(
                id="d_jp",
                name="jp",
                isin="ISIN_JP",
                currency="EUR",
                ter_pct=0.20,
                category="Japan",
                yahoo="x",
            ),
        ]

        # Groups: d_us + d_us_alt are correlated; d_us is the rep (better score)

        groups = [
            GroupRepresentative(
                group=["d_us", "d_us_alt"],
                representative="d_us",
                representative_score=2.0,
                member_scores={"d_us": 2.0, "d_us_alt": 0.5},
            ),
            # d_eu and d_jp in singleton groups (ignored)
            GroupRepresentative(
                group=["d_eu"],
                representative="d_eu",
                representative_score=1.0,
                member_scores={"d_eu": 1.0},
            ),
            GroupRepresentative(
                group=["d_jp"],
                representative="d_jp",
                representative_score=1.0,
                member_scores={"d_jp": 1.0},
            ),
        ]

        return cfg, entries, groups

    def test_redundant_pair_flagged(self) -> None:
        cfg, entries, groups = self._make_setup()
        diagnostics = diagnose_strategies(cfg, entries, groups)

        redundant = [d for d in diagnostics if d.strategy_name == "strat_redundant"]
        assert len(redundant) == 1
        assert redundant[0].issue == "remove"
        assert "us" in redundant[0].detail
        assert "us_alt" in redundant[0].detail

    def test_suboptimal_replace_flagged(self) -> None:
        cfg, entries, groups = self._make_setup()
        diagnostics = diagnose_strategies(cfg, entries, groups)

        suboptimal = [d for d in diagnostics if d.strategy_name == "strat_suboptimal"]
        assert len(suboptimal) == 1
        assert suboptimal[0].issue == "replace"
        assert "us_alt" in suboptimal[0].detail
        assert "d_us" in suboptimal[0].suggestion

    def test_clean_strategy_no_diagnostics(self) -> None:
        cfg, entries, groups = self._make_setup()
        diagnostics = diagnose_strategies(cfg, entries, groups)

        clean = [d for d in diagnostics if d.strategy_name == "strat_clean"]
        assert clean == []

    def test_redundant_strategy_does_not_also_get_replace(self) -> None:
        """If strat_redundant uses both us + us_alt, the 'remove' diagnostic
        already implicitly recommends keeping the rep (us). Don't ALSO emit
        a 'replace us_alt with us' — that's redundant feedback."""

        cfg, entries, groups = self._make_setup()
        diagnostics = diagnose_strategies(cfg, entries, groups)

        redundant_strat_diags = [d for d in diagnostics if d.strategy_name == "strat_redundant"]
        replace_diags = [d for d in redundant_strat_diags if d.issue == "replace"]
        assert replace_diags == []

    def test_strategy_listing_safe_does_not_raise(self) -> None:
        """Regression: under the new rank-only methodology, strategies list
        `safe` directly in `assets:`. `safe` is the synthetic €STR sleeve
        (lives on `config.safe_asset`, not in `config.assets`), so
        `config.asset_by_id("safe")` raises KeyError. The diagnostics loop
        must skip safe — it has no entry in the discovery universe and no
        diagnostics to emit anyway."""
        cfg, entries, groups = self._make_setup()
        # Build a strategy that lists safe alongside risky assets.
        strat_with_safe = Strategy(
            name="strat_with_safe",
            asset_ids=("us", "eu", "safe"),
            rebalance="monthly_first_sunday",
            top_n=2,
        )
        cfg_with_safe = Config(
            shared=cfg.shared,
            assets=cfg.assets,
            strategies=(*cfg.strategies, strat_with_safe),
        )
        # Must not raise.
        diagnostics = diagnose_strategies(cfg_with_safe, entries, groups)
        # The new strategy emits no diagnostics (us is the rep, eu is a
        # singleton group, safe is skipped).
        assert [d for d in diagnostics if d.strategy_name == "strat_with_safe"] == []
