"""Correlation-matrix analysis on the discovery universe.

Three pieces:

1. `pairwise_corrcoef()` — low-level helper: filter, pivot, forward-fill,
   compute pairwise Pearson corrcoef on daily returns. Reused by
   `metrics.avg_pairwise_correlation`.

2. `compute_correlation_matrix()` — wrap `pairwise_corrcoef` over a trailing
   `window_days` window into a `CorrelationMatrix` for rendering.

3. `find_groups()` — union-find on edges where correlation > threshold.
   Connected components are groups of redundant exposures.

4. `best_in_group()` — for each group, picks the representative with the
   best return / cost ratio (CAGR over the window divided by TER).

Strategy diagnostics (cross-referencing strategies.yaml against these
groups) live in `diagnostics.py`.
"""

from __future__ import annotations

import math
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
    """Union-Find groups: assets are in the same group if there's a chain
    of pairwise correlations all above `threshold`. Groups of size 1
    (singletons) are also returned so the caller has a complete partition.

    If `region_by_id` is provided, two assets are unioned only when their
    daily correlation is above `threshold` AND they share the same coarse
    region. This prevents lumping cross-perimeter pairs that share market
    beta but track different universes (e.g. MSCI World vs S&P 500).
    """
    n = len(cm.asset_ids)
    if n == 0:
        return []
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            if cm.matrix[i, j] <= threshold:
                continue
            if region_by_id is not None:
                ri = region_by_id.get(cm.asset_ids[i])
                rj = region_by_id.get(cm.asset_ids[j])
                if ri != rj:
                    continue
            union(i, j)

    groups_by_root: dict[int, list[str]] = {}
    for i in range(n):
        root = find(i)
        groups_by_root.setdefault(root, []).append(cm.asset_ids[i])

    # Sort: largest groups first, then by lead member id for stability.
    return sorted(
        groups_by_root.values(),
        key=lambda g: (-len(g), g[0]),
    )


def best_in_group(
    group: list[str],
    prices_long: pl.DataFrame,
    ter_pct_by_id: Mapping[str, float],
    window_days: int = 252,
) -> GroupRepresentative:
    """Pick the asset with highest score = CAGR - TER (additive net-of-fees
    annual return over the same window). Both CAGR and TER are fractions
    (TER passed in as percent points → divided by 100 internally).

    Single-member groups return that member as representative trivially.
    """
    scores: dict[str, float] = {}
    for asset_id in group:
        cagr = _cagr_over_window(prices_long, asset_id, window_days)
        ter_frac = ter_pct_by_id.get(asset_id, 0.10) / 100.0
        scores[asset_id] = (cagr - ter_frac) if cagr is not None else float("-inf")

    rep = max(scores, key=lambda k: scores[k])
    return GroupRepresentative(
        group=group,
        representative=rep,
        representative_score=scores[rep],
        member_scores=scores,
    )


def _cagr_over_window(prices_long: pl.DataFrame, asset_id: str, window_days: int) -> float | None:
    series = (
        prices_long.filter(pl.col("asset_id") == asset_id)
        .sort("date")
        .tail(window_days + 1)
        .get_column("close")
        .to_list()
    )
    if len(series) < 2 or series[0] <= 0:
        return None
    total_return = series[-1] / series[0] - 1.0
    years = window_days / 252.0
    if years <= 0:
        return None
    return math.copysign(abs(1.0 + total_return) ** (1.0 / years) - 1.0, total_return)
