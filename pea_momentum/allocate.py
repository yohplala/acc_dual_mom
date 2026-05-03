"""Portfolio allocation: rank-only top-N selection + weighting + rounding.

Pipeline at each rebalance:

    raw scores  ──top-N by score──▶              selected
    selected    ──equal_weight or score_prop──▶  raw weights
    raw weights ──largest-remainder rounding──▶  final weights (granularity steps)
    residual    ──▶ residual_holder (caller-provided; safe asset or cash)

Selection is **purely rank-based**: the top-N highest-scoring assets are
chosen regardless of the sign of their scores. There is no positive-
momentum floor — even in a synchronised bear regime where every score
is negative, the strategy still allocates to the least-negative top-N
assets ("hold the leader" rather than "go to cash by filter").

Defensive cash behaviour comes from ranking, not filtering. List
`cash_estr` (or any low-volatility yielding sleeve) in the strategy's
`assets:` and its small positive €STR-driven score will naturally win
the top-N rank when every equity score collapses below zero, producing
a 100%-cash defensive allocation organically. Strategies that don't
list cash forgo this defence and stay equity-allocated through bears.

Two weighting rules are supported:
- ``equal_weight``       1/N across selected assets — canonical Antonacci ADM.
- ``score_proportional`` |score|-proportional weights with rank-based
                         attribution: the highest-scoring asset always
                         gets the largest weight, the lowest-scoring the
                         smallest. For all-positive selections this is the
                         canonical w_i = s_i / Σs_j formula; for mixed-sign
                         or all-negative selections (reachable now that the
                         top-N has no filter) the formula stays well-defined
                         with no negative weights and no sign-flipped ranks.
                         "Least-crashing → highest conviction, sharpest-
                         crashing → lowest conviction" by construction.

Pure function — no I/O, no side effects.
"""

from __future__ import annotations

from .discover import assets_by_region
from .universe import Allocation, Asset

# Sentinel id for un-allocated weight when the strategy has no yielding cash
# sleeve. Treated as 0%-return inside the backtest's compounding kernel (no
# price series → no daily return contribution).
CASH_KEY = "cash"

ALLOCATION_RULE_EQUAL = "equal_weight"
ALLOCATION_RULE_SCORE_PROP = "score_proportional"
SUPPORTED_ALLOCATION_RULES: frozenset[str] = frozenset(
    {ALLOCATION_RULE_EQUAL, ALLOCATION_RULE_SCORE_PROP}
)


def allocate(
    scores: dict[str, float],
    top_n: int,
    alloc: Allocation,
    *,
    rule_override: str | None = None,
    residual_holder: str = CASH_KEY,
    regional_weights: tuple[tuple[str, float], ...] | None = None,
    asset_by_id: dict[str, Asset] | None = None,
) -> dict[str, float]:
    """Return target weights summing to 1.0 across selected assets + residual.

    Three weighting paths:

    - ``equal_weight`` / ``score_proportional`` (default): rank-only top-N
      across all `scores`, then 1/N or |score|-prop with rank attribution.
    - ``regional_weights`` set: dispatch to a static-regional split. Each
      region's selected asset (one per `dashboard_bucket`) gets its
      configured weight; absent regions are forfeited and present-region
      weights renormalise. Score magnitudes are not used — selection has
      already happened upstream (`top_1_per_region`). `asset_by_id` is
      required (region inference needs each asset's `category`).

    All three paths apply the same largest-remainder rounding to integer
    multiples of `alloc.granularity_pct / 100` and route any shortfall to
    `residual_holder`.

    Selection: pick the top-N highest-scoring assets in `scores`, irrespective
    of the sign of their scores. If `scores` is empty (no asset had enough
    history for a signal at this rebalance), the full 100% goes to
    `residual_holder`.

    Weighting: equal-weight or score-proportional, then rounded to integer
    multiples of `alloc.granularity_pct / 100` via largest-remainder. Any
    rounding shortfall goes to `residual_holder`. ``score_proportional``
    uses |score|-magnitudes with rank-based attribution (best score →
    largest weight, worst → smallest), which collapses to the canonical
    w_i = s_i / Σs_j when all selected scores are positive and stays
    well-defined (no shorts, no inverted ranks) for mixed or negative
    sums.

    `residual_holder` is the asset id that catches rounding residuals.
    Typically:
    - the strategy's safe-asset id, if it's listed in the strategy's
      universe → the residual earns €STR yield;
    - or `CASH_KEY` (0%-return placeholder) when the strategy has no
      yielding cash sleeve listed.
    """
    if alloc.rounding != "largest_remainder":
        raise ValueError(f"Unsupported rounding: {alloc.rounding!r}")

    if regional_weights is not None:
        if asset_by_id is None:
            raise ValueError("allocate(regional_weights=...) requires asset_by_id")
        return _allocate_regional_fixed(
            scores=scores,
            asset_by_id=asset_by_id,
            regional_weights=regional_weights,
            granularity_pct=alloc.granularity_pct,
            residual_holder=residual_holder,
        )

    rule = rule_override or alloc.rule
    if rule not in SUPPORTED_ALLOCATION_RULES:
        raise ValueError(
            f"Unsupported allocation rule: {rule!r} "
            f"(supported: {sorted(SUPPORTED_ALLOCATION_RULES)})"
        )

    # Rank-only top-N: keep all scored assets, sort descending, take the head.
    selected = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

    if not selected:
        return {residual_holder: 1.0}

    if rule == ALLOCATION_RULE_EQUAL:
        n = len(selected)
        raw = {a: 1.0 / n for a, _ in selected}
    else:  # score_proportional
        # |score|-proportional weights with rank-based reverse attribution:
        # the highest-scoring asset gets the largest |score|-derived weight,
        # the lowest-scoring gets the smallest. For all-positive selections
        # this collapses to the canonical w_i = s_i / Σs_j formula (because
        # score-rank == |score|-rank when everything is positive). For
        # mixed-sign or all-negative selections (reachable now that the
        # rank-only top-N has no filter), this preserves the
        # "least-crashing → highest conviction" intent without producing
        # negative weights (shorts) or sign-flipped rankings the pure
        # formula yields when sum ≤ 0 or any selected score is negative.
        abs_total = sum(abs(s) for _, s in selected)
        if abs_total == 0:
            # Degenerate: every selected score is exactly zero. Equal-weight
            # is the only well-defined fallback.
            n = len(selected)
            raw = {a: 1.0 / n for a, _ in selected}
        else:
            # `selected` is sorted by score descending (best first); pair
            # each asset with the |score|-magnitudes sorted descending.
            sorted_abs_desc = sorted(abs(s) for _, s in selected)
            sorted_abs_desc.reverse()
            raw = {a: sorted_abs_desc[i] / abs_total for i, (a, _) in enumerate(selected)}

    return _round_to_granularity(
        raw, granularity_pct=alloc.granularity_pct, residual_holder=residual_holder
    )


def _allocate_regional_fixed(
    scores: dict[str, float],
    asset_by_id: dict[str, Asset],
    regional_weights: tuple[tuple[str, float], ...],
    granularity_pct: int,
    residual_holder: str,
) -> dict[str, float]:
    """Static regional split: each region's selected asset gets its
    configured weight, present-region weights renormalise if a region
    is absent, then granularity rounding.

    Inputs (after `_filter_top_1_per_region` upstream): `scores` already
    holds at most one asset per regional bucket — score values are
    ignored here, they only served to break ties inside each region
    upstream. Region inference goes through `dashboard_bucket(category)`,
    so each asset must resolve in `asset_by_id`.

    Missing-region semantics: if a configured region has no asset in
    `scores`, the present regions' weights are renormalised to sum to
    1.0 — the absent region's slot is forfeited, not parked on cash.
    Keeps the strategy fully invested when at least one region has a
    winner. If every region drops out, 100% goes to `residual_holder`.
    """
    region_to_w = dict(regional_weights)
    grouped = assets_by_region(scores, asset_by_id)
    raw: dict[str, float] = {}
    for bucket, bucket_assets in grouped.items():
        w = region_to_w.get(bucket)
        if w is None or w <= 0.0:
            continue
        for asset_id in bucket_assets:
            raw[asset_id] = w
    if not raw:
        return {residual_holder: 1.0}
    total = sum(raw.values())
    if total <= 0.0:
        return {residual_holder: 1.0}
    raw = {a: w / total for a, w in raw.items()}
    return _round_to_granularity(
        raw, granularity_pct=granularity_pct, residual_holder=residual_holder
    )


def _round_to_granularity(
    raw: dict[str, float], granularity_pct: int, *, residual_holder: str
) -> dict[str, float]:
    """Largest-remainder (Hare) rounding to integer multiples of granularity_pct.

    Total target is 100%. Any shortfall after risky-asset rounding goes to
    `residual_holder` (typically the safe asset id, or CASH_KEY).
    """
    if granularity_pct <= 0 or 100 % granularity_pct != 0:
        raise ValueError(f"granularity_pct must divide 100; got {granularity_pct}")
    total_steps = 100 // granularity_pct

    scaled = {a: w * total_steps for a, w in raw.items()}
    floored = {a: int(v) for a, v in scaled.items()}
    used = sum(floored.values())
    remaining = total_steps - used

    remainders = sorted(
        ((a, scaled[a] - floored[a]) for a in scaled),
        key=lambda kv: kv[1],
        reverse=True,
    )
    for asset_id, _ in remainders[:remaining]:
        floored[asset_id] += 1
    risky_steps_total = sum(floored.values())
    residual_steps = total_steps - risky_steps_total

    weights: dict[str, float] = {
        a: steps * granularity_pct / 100.0 for a, steps in floored.items() if steps > 0
    }
    if residual_steps > 0:
        # If the residual_holder happens to also be a selected asset (safe
        # was both ranked top-N and would catch residual), merge into its
        # existing weight rather than duplicating the key.
        residual_weight = residual_steps * granularity_pct / 100.0
        weights[residual_holder] = weights.get(residual_holder, 0.0) + residual_weight
    return weights
