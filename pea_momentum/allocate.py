"""Portfolio allocation: absolute-momentum filter, top-N selection, weighting.

Pipeline at each rebalance:

    raw scores  ──filter(score > safe_score)──▶  candidates
    candidates  ──top-N by score──▶              selected
    selected    ──score-proportional──▶          raw weights
    raw weights ──largest-remainder rounding──▶  final weights (10% steps)
    residual to safe asset

Pure function — no I/O, no side effects. The backtest calls this for each
rebalance day and feeds it the score dict.
"""

from __future__ import annotations

from .universe import Allocation, Filter

SAFE_ASSET_KEY = "safe"


def allocate(
    scores: dict[str, float],
    safe_score: float,
    top_n: int,
    alloc: Allocation,
    flt: Filter,
) -> dict[str, float]:
    """Return target weights summing to 1.0 across selected assets + safe asset.

    Weights are integer multiples of `alloc.granularity_pct / 100`.
    Anything not allocated to risky assets goes to `safe`.
    """
    if alloc.rule != "score_proportional":
        raise ValueError(f"Unsupported allocation rule: {alloc.rule!r}")
    if alloc.rounding != "largest_remainder":
        raise ValueError(f"Unsupported rounding: {alloc.rounding!r}")
    if flt.type != "absolute_momentum" or flt.benchmark != "safe_asset":
        raise ValueError(f"Unsupported filter: {flt!r}")

    candidates = {a: s for a, s in scores.items() if s > safe_score and s > 0}
    selected = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

    if not selected:
        return {SAFE_ASSET_KEY: 1.0}

    total = sum(s for _, s in selected)
    raw = {a: s / total for a, s in selected}

    return _round_to_granularity(raw, granularity_pct=alloc.granularity_pct)


def _round_to_granularity(raw: dict[str, float], granularity_pct: int) -> dict[str, float]:
    """Largest-remainder (Hare) rounding to integer multiples of granularity_pct.

    Total target is 100%. Any shortfall after risky-asset rounding goes to `safe`.
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
    safe_steps = total_steps - risky_steps_total

    weights: dict[str, float] = {
        a: steps * granularity_pct / 100.0 for a, steps in floored.items() if steps > 0
    }
    if safe_steps > 0:
        weights[SAFE_ASSET_KEY] = safe_steps * granularity_pct / 100.0
    return weights
