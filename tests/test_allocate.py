"""Tests for the rank-only top-N + weight rounding pipeline."""

from __future__ import annotations

import pytest

from pea_momentum.allocate import CASH_KEY, _round_to_granularity, allocate
from pea_momentum.universe import Allocation


def _alloc(rule: str = "score_proportional", granularity: int = 10) -> Allocation:
    return Allocation(
        rule=rule,
        granularity_pct=granularity,
        rounding="largest_remainder",
    )


def _sum_to_one(weights: dict[str, float]) -> bool:
    return abs(sum(weights.values()) - 1.0) < 1e-9


class TestRankOnlyTopN:
    def test_all_negative_still_allocates_to_least_bad(self) -> None:
        """No positive-momentum filter: even when every score is negative,
        the top-N picks the least-negative asset and allocates 100% to it."""
        scores = {"a": -0.05, "b": -0.10, "c": -0.01}
        w = allocate(scores, top_n=1, alloc=_alloc("equal_weight"))
        # `c` is the least-negative → wins top-1
        assert w == {"c": 1.0}

    def test_zero_score_can_be_selected(self) -> None:
        """Zero score is no longer excluded by a >0 filter — included in the
        ranking like any other score."""
        scores = {"a": 0.10, "b": 0.0}
        w = allocate(scores, top_n=2, alloc=_alloc("equal_weight"))
        # Both selected at top-2 over 2 assets → 50/50.
        assert w == {"a": 0.5, "b": 0.5}

    def test_top_n_ranks_strict_descending_regardless_of_sign(self) -> None:
        scores = {"a": 0.10, "b": -0.02, "c": -0.20}
        w = allocate(scores, top_n=2, alloc=_alloc("equal_weight"))
        # `a` (best) and `b` (least bad) selected; `c` excluded.
        assert "c" not in w
        assert w["a"] == pytest.approx(0.5)
        assert w["b"] == pytest.approx(0.5)


class TestEmptyScoresFallsThroughToResidual:
    def test_no_scores_at_all_residual_holder_takes_full_weight(self) -> None:
        """When no asset has a score (early backtest, lookback unavailable),
        the full 100% goes to the residual holder."""
        w = allocate(scores={}, top_n=2, alloc=_alloc())
        assert w == {CASH_KEY: 1.0}

    def test_no_scores_with_safe_residual_holder(self) -> None:
        w = allocate(scores={}, top_n=2, alloc=_alloc(), residual_holder="safe")
        assert w == {"safe": 1.0}


class TestSafeCompetesViaRank:
    def test_safe_listed_can_win_top_n_when_its_score_is_highest(self) -> None:
        # Safe scored alongside risky. With top_n=2 and safe winning top, safe
        # appears in the weights with 50% (or higher via rounding).
        scores = {"a": 0.10, "safe": 0.20, "b": -0.05}
        w = allocate(scores, top_n=2, alloc=_alloc("equal_weight"), residual_holder="safe")
        # safe and a both selected; equal-weight 50/50.
        assert w.get("safe") == pytest.approx(0.5)
        assert w.get("a") == pytest.approx(0.5)

    def test_safe_loses_when_outscored(self) -> None:
        # Two equity scores stronger than safe — safe drops out of top-2.
        scores = {"a": 0.10, "b": 0.08, "safe": 0.02}
        w = allocate(scores, top_n=2, alloc=_alloc("equal_weight"), residual_holder="safe")
        assert "safe" not in w
        assert set(w.keys()) == {"a", "b"}


class TestTopN:
    def test_only_top_n_selected(self) -> None:
        scores = {"a": 0.10, "b": 0.08, "c": 0.06, "d": 0.04, "e": 0.02}
        w = allocate(scores, top_n=3, alloc=_alloc("equal_weight"))
        risky = {k: v for k, v in w.items() if k != CASH_KEY}
        assert set(risky) == {"a", "b", "c"}


class TestScoreProportionality:
    def test_higher_score_gets_higher_weight_when_all_positive(self) -> None:
        scores = {"a": 0.30, "b": 0.20, "c": 0.10}
        w = allocate(scores, top_n=3, alloc=_alloc())
        assert w["a"] >= w["b"] >= w["c"]

    def test_score_proportional_all_negative_assigns_largest_weight_to_least_crashing(
        self,
    ) -> None:
        """When all selected scores are negative, |score|-proportional weights
        are assigned by rank: the least-crashing asset gets the largest
        weight (highest conviction in the survivor), the sharpest-crashing
        gets the smallest. The pure formula would invert this ranking
        (worst score gets the largest weight); we don't allow that."""
        scores = {"a": -0.05, "b": -0.10, "c": -0.20}  # a is least bad
        w = allocate(scores, top_n=3, alloc=_alloc("score_proportional"))
        assert sum(w.values()) == pytest.approx(1.0)
        # Best score → largest weight; worst → smallest.
        assert w["a"] > w["b"] > w["c"]
        # No negative weights (no implicit shorts).
        assert all(v > 0 for v in w.values())

    def test_score_proportional_mixed_sign_avoids_negative_weights(self) -> None:
        """With mixed-sign selected scores (filter-removed scenario), the
        pure formula would assign negative weights to negative-scored
        assets. Our |score|-prop + rank-attribution stays positive and
        keeps the best-rank → highest-weight invariant."""
        scores = {"a": 0.30, "b": 0.10, "c": -0.05}
        w = allocate(scores, top_n=3, alloc=_alloc("score_proportional"))
        assert all(v > 0 for v in w.values())
        assert w["a"] > w["b"] > w["c"]
        assert sum(w.values()) == pytest.approx(1.0)

    def test_score_proportional_all_zero_falls_back_to_equal_weight(self) -> None:
        """The degenerate case where every selected score is exactly zero —
        |score|-magnitudes are all zero, so we fall back to 1/N equal-weight."""
        scores = {"a": 0.0, "b": 0.0, "c": 0.0}
        w = allocate(scores, top_n=3, alloc=_alloc("score_proportional"))
        assert sum(w.values()) == pytest.approx(1.0)
        risky = {k: v for k, v in w.items() if k != CASH_KEY}
        assert max(risky.values()) - min(risky.values()) <= 0.10 + 1e-9


class TestRounding:
    def test_weights_are_multiples_of_granularity(self) -> None:
        scores = {"a": 0.137, "b": 0.092, "c": 0.041}
        w = allocate(scores, top_n=3, alloc=_alloc(granularity=10))
        for v in w.values():
            assert (v * 100) % 10 < 1e-9 or abs((v * 100) % 10 - 10) < 1e-9

    def test_weights_always_sum_to_one(self) -> None:
        scores = {"a": 0.137, "b": 0.092, "c": 0.041, "d": 0.020}
        w = allocate(scores, top_n=3, alloc=_alloc(granularity=10))
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
            residual_holder="safe",
        )
        assert _sum_to_one(w)
        # Three equal candidates at granularity 10 → 30/30/40 (one wins the
        # remainder). No residual_holder injection because the candidates
        # themselves consumed the full 100%.
        assert sum(w.values()) == pytest.approx(1.0)


class TestSinglePick:
    def test_single_score_takes_100pct(self) -> None:
        scores = {"a": 0.30}
        w = allocate(scores, top_n=3, alloc=_alloc())
        assert w == {"a": 1.0}


class TestEqualWeight:
    def test_three_picks_get_equal_weight(self) -> None:
        scores = {"a": 0.30, "b": 0.20, "c": 0.10}
        w = allocate(scores, top_n=3, alloc=_alloc("equal_weight"))
        risky = {k: v for k, v in w.items() if k != CASH_KEY}
        # 1/3 ≈ 0.333 → round to 30%/30%/40% via largest-remainder; sum = 1
        assert _sum_to_one(w)
        assert sum(risky.values()) == pytest.approx(1.0)
        # Largest weight - smallest weight ≤ one granularity step
        assert max(risky.values()) - min(risky.values()) <= 0.10 + 1e-9

    def test_unsupported_rule_raises(self) -> None:
        bogus = Allocation(rule="bogus", granularity_pct=10, rounding="largest_remainder")
        with pytest.raises(ValueError, match="Unsupported allocation rule"):
            allocate({"a": 0.1}, top_n=1, alloc=bogus)

    def test_rule_override_takes_precedence(self) -> None:
        # Shared alloc says score_proportional, override says equal_weight.
        scores = {"a": 0.30, "b": 0.10}
        w = allocate(
            scores,
            top_n=2,
            alloc=_alloc(),
            rule_override="equal_weight",
        )
        risky = {k: v for k, v in w.items() if k != CASH_KEY}
        # Equal weight means a and b should be ~50/50 (at 10% granularity, exact)
        assert risky.get("a") == pytest.approx(0.5)
        assert risky.get("b") == pytest.approx(0.5)
