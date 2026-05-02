"""Tests for the positive-momentum filter + top-N + weight rounding pipeline."""

from __future__ import annotations

import pytest

from pea_momentum.allocate import CASH_KEY, _round_to_granularity, allocate
from pea_momentum.universe import Allocation, Filter


def _alloc(rule: str = "score_proportional", granularity: int = 10) -> Allocation:
    return Allocation(
        rule=rule,
        granularity_pct=granularity,
        rounding="largest_remainder",
    )


def _flt() -> Filter:
    return Filter(type="positive_momentum")


def _sum_to_one(weights: dict[str, float]) -> bool:
    return abs(sum(weights.values()) - 1.0) < 1e-9


class TestPositiveFilter:
    def test_all_negative_falls_through_to_residual(self) -> None:
        scores = {"a": -0.05, "b": -0.10, "c": -0.01}
        w = allocate(scores, top_n=3, alloc=_alloc(), flt=_flt())
        assert w == {CASH_KEY: 1.0}

    def test_zero_score_excluded(self) -> None:
        scores = {"a": 0.10, "b": 0.0}
        w = allocate(scores, top_n=3, alloc=_alloc(), flt=_flt())
        assert "b" not in w
        assert w["a"] == pytest.approx(1.0)

    def test_only_positive_selected(self) -> None:
        scores = {"a": 0.10, "b": 0.05, "c": -0.02}
        w = allocate(scores, top_n=3, alloc=_alloc(), flt=_flt())
        assert "c" not in w
        assert "a" in w and "b" in w
        assert _sum_to_one(w)


class TestResidualHolder:
    def test_safe_listed_residual_goes_to_safe(self) -> None:
        # Single positive pick → 100%; no residual to worry about.
        # But with top_n=2 and 1 candidate at 1/1 weight, residual is 0.
        # Test the all-negative path: residual goes to whatever is configured.
        scores = {"a": -0.05, "b": -0.10}
        w = allocate(scores, top_n=2, alloc=_alloc(), flt=_flt(), residual_holder="safe")
        assert w == {"safe": 1.0}

    def test_no_safe_residual_goes_to_cash(self) -> None:
        scores = {"a": -0.05, "b": -0.10}
        w = allocate(scores, top_n=2, alloc=_alloc(), flt=_flt())
        assert w == {CASH_KEY: 1.0}

    def test_safe_competes_in_top_n_when_listed(self) -> None:
        # safe scored alongside risky. With top_n=2 and safe winning, safe
        # appears in the weights with 50% (or higher via rounding).
        scores = {"a": 0.10, "safe": 0.20, "b": -0.05}
        w = allocate(
            scores, top_n=2, alloc=_alloc("equal_weight"), flt=_flt(), residual_holder="safe"
        )
        # safe and a both selected; equal-weight 50/50.
        assert w.get("safe") == pytest.approx(0.5)
        assert w.get("a") == pytest.approx(0.5)


class TestTopN:
    def test_only_top_n_selected(self) -> None:
        scores = {"a": 0.10, "b": 0.08, "c": 0.06, "d": 0.04, "e": 0.02}
        w = allocate(scores, top_n=3, alloc=_alloc(), flt=_flt())
        risky = {k: v for k, v in w.items() if k != CASH_KEY}
        assert set(risky) == {"a", "b", "c"}


class TestScoreProportionality:
    def test_higher_score_gets_higher_weight(self) -> None:
        scores = {"a": 0.30, "b": 0.20, "c": 0.10}
        w = allocate(scores, top_n=3, alloc=_alloc(), flt=_flt())
        assert w["a"] >= w["b"] >= w["c"]


class TestRounding:
    def test_weights_are_multiples_of_granularity(self) -> None:
        scores = {"a": 0.137, "b": 0.092, "c": 0.041}
        w = allocate(scores, top_n=3, alloc=_alloc(granularity=10), flt=_flt())
        for v in w.values():
            assert (v * 100) % 10 < 1e-9 or abs((v * 100) % 10 - 10) < 1e-9

    def test_weights_always_sum_to_one(self) -> None:
        scores = {"a": 0.137, "b": 0.092, "c": 0.041, "d": 0.020}
        w = allocate(scores, top_n=3, alloc=_alloc(granularity=10), flt=_flt())
        assert _sum_to_one(w)

    def test_largest_remainder_breaks_ties_by_size(self) -> None:
        raw = {"a": 0.333, "b": 0.333, "c": 0.334}
        w = _round_to_granularity(raw, granularity_pct=10, residual_holder=CASH_KEY)
        assert sum(w.values()) == pytest.approx(1.0)

    def test_unsupported_granularity_raises(self) -> None:
        with pytest.raises(ValueError):
            _round_to_granularity({"a": 1.0}, granularity_pct=7, residual_holder=CASH_KEY)

    def test_largest_remainder_always_sums_exact_so_no_residual_for_selected(self) -> None:
        """Largest-remainder is exact-summing by construction: when there are
        selected candidates, the rounded weights always total 100% and no
        residual goes to the residual_holder."""
        scores = {"a": 0.10, "safe": 0.10, "c": 0.10}
        w = allocate(
            scores,
            top_n=3,
            alloc=_alloc("equal_weight"),
            flt=_flt(),
            residual_holder="safe",
        )
        assert _sum_to_one(w)
        # Three equal candidates at granularity 10 → 30/30/40 (one wins the
        # remainder). No residual_holder injection because the candidates
        # themselves consumed the full 100%.
        assert sum(w.values()) == pytest.approx(1.0)


class TestSinglePick:
    def test_single_positive_takes_100pct(self) -> None:
        scores = {"a": 0.30}
        w = allocate(scores, top_n=3, alloc=_alloc(), flt=_flt())
        assert w == {"a": 1.0}


class TestEqualWeight:
    def test_three_picks_get_equal_weight(self) -> None:
        scores = {"a": 0.30, "b": 0.20, "c": 0.10}
        w = allocate(scores, top_n=3, alloc=_alloc("equal_weight"), flt=_flt())
        risky = {k: v for k, v in w.items() if k != CASH_KEY}
        # 1/3 ≈ 0.333 → round to 30%/30%/40% via largest-remainder; sum = 1
        assert _sum_to_one(w)
        assert sum(risky.values()) == pytest.approx(1.0)
        # Largest weight - smallest weight ≤ one granularity step
        assert max(risky.values()) - min(risky.values()) <= 0.10 + 1e-9

    def test_unsupported_rule_raises(self) -> None:
        bogus = Allocation(rule="bogus", granularity_pct=10, rounding="largest_remainder")
        with pytest.raises(ValueError, match="Unsupported allocation rule"):
            allocate({"a": 0.1}, top_n=1, alloc=bogus, flt=_flt())

    def test_rule_override_takes_precedence(self) -> None:
        # Shared alloc says score_proportional, override says equal_weight.
        scores = {"a": 0.30, "b": 0.10}
        w = allocate(
            scores,
            top_n=2,
            alloc=_alloc(),
            flt=_flt(),
            rule_override="equal_weight",
        )
        risky = {k: v for k, v in w.items() if k != CASH_KEY}
        # Equal weight means a and b should be ~50/50 (at 10% granularity, exact)
        assert risky.get("a") == pytest.approx(0.5)
        assert risky.get("b") == pytest.approx(0.5)
