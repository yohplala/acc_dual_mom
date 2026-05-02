"""Portfolio allocation: positive-momentum filter, top-N selection, weighting.

Pipeline at each rebalance:

    raw scores  ──filter(score > 0)──▶          candidates
    candidates  ──top-N by score──▶              selected
    selected    ──equal_weight or score_prop──▶  raw weights
    raw weights ──largest-remainder rounding──▶  final weights (granularity steps)
    residual    ──▶ residual_holder (caller-provided; safe asset or cash)

Two weighting rules are supported:
- ``equal_weight``       1/N across selected assets — canonical Antonacci ADM.
- ``score_proportional`` w_i = s_i / Σs_j over selected — amplifies the loudest
                         score; useful as a sensitivity exhibit.

Note: this module no longer auto-injects a "safe" sentinel. The caller decides
which asset id catches rounding residuals (typically the safe asset if listed
in the strategy's universe, otherwise a 0%-return CASH placeholder). Whether
the safe asset participates in the score → top-N ranking is also the caller's
choice — list it in the strategy's `assets:` and it competes naturally.

Pure function — no I/O, no side effects.
"""

from __future__ import annotations

from .universe import Allocation, Filter

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
    flt: Filter,
    *,
    rule_override: str | None = None,
    residual_holder: str = CASH_KEY,
) -> dict[str, float]:
    """Return target weights summing to 1.0 across selected assets + residual.

    Selection: from the assets in `scores` whose score exceeds 0 (the
    positive-momentum floor), pick the top-N highest. If none qualify, the
    full 100% goes to `residual_holder`.

    Weighting: equal-weight or score-proportional, then rounded to integer
    multiples of `alloc.granularity_pct / 100` via largest-remainder. Any
    rounding shortfall goes to `residual_holder`.

    `residual_holder` is the asset id that catches both the "no candidate"
    fallback and rounding residuals. Typically:
    - the strategy's safe-asset id, if `safe` is listed in the strategy's
      universe → the residual earns €STR yield;
    - or `CASH_KEY` (0%-return placeholder) when the strategy has no
      yielding cash sleeve listed.
    """
    rule = rule_override or alloc.rule
    if rule not in SUPPORTED_ALLOCATION_RULES:
        raise ValueError(
            f"Unsupported allocation rule: {rule!r} "
            f"(supported: {sorted(SUPPORTED_ALLOCATION_RULES)})"
        )
    if alloc.rounding != "largest_remainder":
        raise ValueError(f"Unsupported rounding: {alloc.rounding!r}")
    if flt.type != "positive_momentum":
        raise ValueError(f"Unsupported filter: {flt.type!r} (expected 'positive_momentum')")

    candidates = {a: s for a, s in scores.items() if s > 0}
    selected = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

    if not selected:
        return {residual_holder: 1.0}

    if rule == ALLOCATION_RULE_EQUAL:
        n = len(selected)
        raw = {a: 1.0 / n for a, _ in selected}
    else:  # score_proportional
        total = sum(s for _, s in selected)
        raw = {a: s / total for a, s in selected}

    return _round_to_granularity(
        raw, granularity_pct=alloc.granularity_pct, residual_holder=residual_holder
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
