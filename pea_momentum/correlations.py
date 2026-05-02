"""Correlation-matrix analysis on the discovery universe.

Pieces:

1. `pairwise_corrcoef()` — low-level helper: filter, pivot, forward-fill,
   compute pairwise Pearson corrcoef on daily returns. Reused by
   `metrics.avg_pairwise_correlation`.

2. `compute_correlation_matrix()` — wrap `pairwise_corrcoef` over a trailing
   `window_days` window into a `CorrelationMatrix` for rendering.

3. `find_groups()` — complete-link agglomeration on edges where correlation
   exceeds `threshold`. Two assets share a group iff every member-to-member
   correlation in the group exceeds the threshold (no transitive chaining
   through medium-correlated bridge pairs).

4. `best_in_group()` — for each group, picks the representative with the
   lowest TER. TER is decided ex-ante and is not subject to in-sample
   selection bias unlike a 1-year-CAGR ranking.

Strategy diagnostics (cross-referencing strategies.yaml against these
groups) live in `diagnostics.py`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True, slots=True)
class CorrelationMatrix:
    asset_ids: list[str]
    matrix: np.ndarray  # n-by-n, symmetric, diag = 1
    window_days: int


@dataclass(frozen=True, slots=True)
class GroupRepresentative:
    group: list[str]
    representative: str
    representative_score: float
    member_scores: dict[str, float]


def pairwise_corrcoef(
    prices_long: pl.DataFrame,
    asset_ids: list[str],
    window_days: int | None = None,
) -> tuple[list[str], np.ndarray] | None:
    """Filter, pivot, forward-fill, and compute pairwise Pearson corrcoef of
    daily returns. Returns `(column_order, n-by-n matrix)` or `None` if
    insufficient data (fewer than 2 assets or fewer than 2 returns).

    `window_days = None` uses the full available history; otherwise only the
    trailing `window_days + 1` rows are used.
    """
    relevant = prices_long.filter(pl.col("asset_id").is_in(asset_ids))
    if relevant.is_empty():
        return None
    wide = relevant.pivot(values="close", index="date", on="asset_id").sort("date")
    cols = [c for c in wide.columns if c != "date"]
    if len(cols) < 2:
        return None
    wide = wide.with_columns([pl.col(c).forward_fill() for c in cols])
    if window_days is not None:
        wide = wide.tail(window_days + 1)
    rets = wide.select(
        *[(pl.col(c) / pl.col(c).shift(1) - 1.0).alias(c) for c in cols]
    ).drop_nulls()
    if rets.height < 2:
        return None
    arr = rets.to_numpy()
    corr = np.corrcoef(arr, rowvar=False)
    if corr.ndim == 0 or corr.shape[0] < 2:
        return None
    return cols, corr


def compute_correlation_matrix(
    prices_long: pl.DataFrame,
    asset_ids: list[str],
    window_days: int = 252,
) -> CorrelationMatrix:
    """Pairwise Pearson correlation of daily returns over the trailing
    `window_days` business days available in `prices_long`.

    Asset ids missing from the data are silently dropped from the result.
    Returns an empty matrix when fewer than two assets have usable history.
    """
    result = pairwise_corrcoef(prices_long, asset_ids, window_days=window_days)
    if result is None:
        return CorrelationMatrix(asset_ids=[], matrix=np.empty((0, 0)), window_days=window_days)
    cols, corr = result
    return CorrelationMatrix(asset_ids=cols, matrix=corr, window_days=window_days)


def find_groups(
    cm: CorrelationMatrix,
    threshold: float = 0.90,
    region_by_id: Mapping[str, str] | None = None,
) -> list[list[str]]:
    """Complete-link agglomeration: two assets share a group iff every
    member-to-member correlation in the group exceeds `threshold`. This
    avoids the chaining property of single-link clustering, where assets
    A and C end up in the same group via a bridge pair (A-B = 0.91,
    B-C = 0.91, A-C = 0.78). For "drop redundant exposure" recommendations,
    we want the conservative complete-link reading.

    Groups of size 1 (singletons) are returned so the caller has a complete
    partition.

    If `region_by_id` is provided, an asset only joins a group when its
    pairwise correlation with EVERY existing member is above `threshold`
    AND it shares the same coarse region. This prevents lumping cross-
    perimeter pairs that share market beta but track different universes
    (e.g. MSCI World vs S&P 500).
    """
    n = len(cm.asset_ids)
    if n == 0:
        return []

    # Greedy complete-link: walk the asset list once, assign each asset to
    # the first group whose every existing member crosses the threshold
    # (and shares a region if region_by_id is provided), or open a new
    # singleton group if no group is fully compatible.
    groups: list[list[int]] = []  # list of indices
    for i in range(n):
        ri = region_by_id.get(cm.asset_ids[i]) if region_by_id is not None else None
        joined = False
        for grp in groups:
            if region_by_id is not None:
                rj = region_by_id.get(cm.asset_ids[grp[0]])
                if ri != rj:
                    continue
            if all(cm.matrix[i, j] > threshold for j in grp):
                grp.append(i)
                joined = True
                break
        if not joined:
            groups.append([i])

    out = [[cm.asset_ids[i] for i in grp] for grp in groups]
    # Sort: largest groups first, then by lead member id for stability.
    return sorted(out, key=lambda g: (-len(g), g[0]))


def best_in_group(
    group: list[str],
    ter_pct_by_id: Mapping[str, float],
) -> GroupRepresentative:
    """Pick the lowest-TER member of `group`. TER is decided ex-ante and is
    immune to the in-sample selection bias that ranking by 1-year CAGR
    introduces (effectively momentum-on-momentum, defeating the purpose
    of dual-momentum's prospective filter).

    Score is `-ter_pct` so that higher = better, matching the rest of the
    representative-score plumbing. Members without an explicit TER use
    `float("inf")` (worst) — they should always lose to a documented
    alternative.

    Single-member groups return that member as representative trivially.
    """
    scores: dict[str, float] = {
        asset_id: -ter_pct_by_id.get(asset_id, float("inf")) for asset_id in group
    }
    rep = max(scores, key=lambda k: scores[k])
    return GroupRepresentative(
        group=group,
        representative=rep,
        representative_score=scores[rep],
        member_scores=scores,
    )
