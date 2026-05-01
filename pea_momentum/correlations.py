"""Correlation-matrix analysis on the discovery universe.

Three pieces:

1. `compute_correlation_matrix()` — pairwise Pearson correlation of daily
   returns over a trailing window. Returns ordered asset ids + the n-by-n
   matrix as a numpy array.

2. `find_groups()` — union-find on edges where correlation > threshold.
   Connected components are groups of redundant exposures.

3. `best_in_group()` — for each group, picks the representative with the
   best return / cost ratio (CAGR over the window divided by TER).
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


def compute_correlation_matrix(
    prices_long: pl.DataFrame,
    asset_ids: list[str],
    window_days: int = 252,
) -> CorrelationMatrix:
    """Pairwise Pearson correlation of daily returns over the trailing
    `window_days` business days available in `prices_long`.

    Asset ids missing from the data are silently dropped from the result —
    not all of `asset_ids` will appear in `matrix.asset_ids` if some
    weren't fetched successfully.
    """
    relevant = prices_long.filter(pl.col("asset_id").is_in(asset_ids))
    if relevant.is_empty():
        return CorrelationMatrix(asset_ids=[], matrix=np.empty((0, 0)), window_days=window_days)

    wide = relevant.pivot(values="close", index="date", on="asset_id").sort("date")
    cols = [c for c in wide.columns if c != "date"]
    if len(cols) < 2:
        return CorrelationMatrix(asset_ids=cols, matrix=np.eye(len(cols)), window_days=window_days)

    wide = wide.with_columns([pl.col(c).forward_fill() for c in cols]).tail(window_days + 1)
    rets = wide.select(
        *[(pl.col(c) / pl.col(c).shift(1) - 1.0).alias(c) for c in cols]
    ).drop_nulls()
    if rets.height < 2:
        return CorrelationMatrix(asset_ids=cols, matrix=np.eye(len(cols)), window_days=window_days)

    arr = rets.to_numpy()
    corr = np.corrcoef(arr, rowvar=False)
    if corr.ndim == 0:
        corr = np.array([[1.0]])
    return CorrelationMatrix(asset_ids=cols, matrix=corr, window_days=window_days)


def find_groups(
    cm: CorrelationMatrix,
    threshold: float = 0.85,
) -> list[list[str]]:
    """Union-Find groups: assets are in the same group if there's a chain
    of pairwise correlations all above `threshold`. Groups of size 1
    (singletons) are also returned so the caller has a complete partition.
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
            if cm.matrix[i, j] > threshold:
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
    """Pick the asset with highest score = CAGR / max(TER, 0.01).

    Single-member groups return that member as representative trivially.
    """
    scores: dict[str, float] = {}
    for asset_id in group:
        cagr = _cagr_over_window(prices_long, asset_id, window_days)
        ter = max(ter_pct_by_id.get(asset_id, 0.10), 0.01)  # avoid div-by-zero
        scores[asset_id] = cagr / ter if cagr is not None else float("-inf")

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
