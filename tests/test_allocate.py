"""Tests for the absolute-momentum filter + top-N + weight rounding pipeline."""

from __future__ import annotations

import pytest

from pea_momentum.allocate import SAFE_ASSET_KEY, _round_to_granularity, allocate
from pea_momentum.universe import Allocation, Filter


def _alloc(granularity: int = 10) -> Allocation:
    return Allocation(
        rule="score_proportional",
        granularity_pct=granularity,
        min_weight_pct=0,
        rounding="largest_remainder",
    )


def _flt() -> Filter:
    return Filter(type="absolute_momentum", benchmark="safe_asset")


def _sum_to_one(weights: dict[str, float]) -> bool:
    return abs(sum(weights.values()) - 1.0) < 1e-9


class TestAbsoluteFilter:
    def test_all_below_safe_goes_to_cash(self) -> None:
        scores = {"a": -0.05, "b": -0.10, "c": 0.00}
        w = allocate(scores, safe_score=0.01, top_n=3, alloc=_alloc(), flt=_flt())
        assert w == {SAFE_ASSET_KEY: 1.0}

    def test_only_above_safe_selected(self) -> None:
        scores = {"a": 0.10, "b": 0.05, "c": -0.02}
        w = allocate(scores, safe_score=0.00, top_n=3, alloc=_alloc(), flt=_flt())
        assert "c" not in w
        assert "a" in w and "b" in w
        assert _sum_to_one(w)

    def test_negative_score_excluded_even_if_above_safe(self) -> None:
        scores = {"a": 0.10, "b": -0.01}
        w = allocate(scores, safe_score=-0.05, top_n=3, alloc=_alloc(), flt=_flt())
        assert "b" not in w


class TestTopN:
    def test_only_top_n_selected(self) -> None:
        scores = {"a": 0.10, "b": 0.08, "c": 0.06, "d": 0.04, "e": 0.02}
        w = allocate(scores, safe_score=0.00, top_n=3, alloc=_alloc(), flt=_flt())
        risky = {k: v for k, v in w.items() if k != SAFE_ASSET_KEY}
        assert set(risky) == {"a", "b", "c"}


class TestScoreProportionality:
    def test_higher_score_gets_higher_weight(self) -> None:
        scores = {"a": 0.30, "b": 0.20, "c": 0.10}
        w = allocate(scores, safe_score=0.00, top_n=3, alloc=_alloc(), flt=_flt())
        assert w["a"] >= w["b"] >= w["c"]

    def test_equal_scores_yield_close_weights(self) -> None:
        scores = {"a": 0.10, "b": 0.10, "c": 0.10}
        w = allocate(scores, safe_score=0.00, top_n=3, alloc=_alloc(), flt=_flt())
        risky = {k: v for k, v in w.items() if k != SAFE_ASSET_KEY}
        assert max(risky.values()) - min(risky.values()) <= 0.10 + 1e-9


class TestRounding:
    def test_weights_are_multiples_of_granularity(self) -> None:
        scores = {"a": 0.137, "b": 0.092, "c": 0.041}
        w = allocate(scores, safe_score=0.00, top_n=3, alloc=_alloc(10), flt=_flt())
        for v in w.values():
            assert (v * 100) % 10 < 1e-9 or abs((v * 100) % 10 - 10) < 1e-9

    def test_weights_always_sum_to_one(self) -> None:
        scores = {"a": 0.137, "b": 0.092, "c": 0.041, "d": 0.020}
        w = allocate(scores, safe_score=0.00, top_n=3, alloc=_alloc(10), flt=_flt())
        assert _sum_to_one(w)

    def test_largest_remainder_breaks_ties_by_size(self) -> None:
        raw = {"a": 0.333, "b": 0.333, "c": 0.334}
        w = _round_to_granularity(raw, granularity_pct=10)
        assert sum(w.values()) == pytest.approx(1.0)

    def test_unsupported_granularity_raises(self) -> None:
        with pytest.raises(ValueError):
            _round_to_granularity({"a": 1.0}, granularity_pct=7)


class TestSafeAssetResidual:
    def test_single_risky_takes_100pct(self) -> None:
        scores = {"a": 0.30}
        w = allocate(scores, safe_score=0.00, top_n=3, alloc=_alloc(), flt=_flt())
        assert w == {"a": 1.0}
        assert SAFE_ASSET_KEY not in w

    def test_two_picks_round_to_full(self) -> None:
        scores = {"a": 0.10, "b": 0.10}
        w = allocate(scores, safe_score=0.00, top_n=3, alloc=_alloc(), flt=_flt())
        risky = sum(v for k, v in w.items() if k != SAFE_ASSET_KEY)
        assert risky == pytest.approx(1.0)
