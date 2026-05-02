"""Accelerated dual-momentum scoring.

For each asset, score(t) = aggregate over lookbacks of (close(t) / close(t - L) - 1).
Computed on EUR-denominated ETF closes (or the €STR-derived synthetic for the
safe asset). Output is a vector of scores indexed by asset_id, evaluated at a
single signal date.
"""

from __future__ import annotations

import statistics
from datetime import date

import numpy as np
import polars as pl

from .universe import Scoring


def score_at(
    prices_long: pl.DataFrame,
    asset_ids: list[str],
    as_of: date,
    cfg: Scoring,
) -> dict[str, float]:
    """Compute scores for `asset_ids` at `as_of` using `cfg.lookbacks_days`.

    Missing assets or insufficient history yield no entry in the output. The
    backtest treats missing scores as "ineligible this period".

    Vectorised across assets via a single pivot + numpy-array indexing.
    """
    if cfg.aggregation != "mean":
        raise ValueError(f"Only aggregation=mean is supported (got {cfg.aggregation!r})")

    relevant = prices_long.filter(pl.col("asset_id").is_in(asset_ids) & (pl.col("date") <= as_of))
    if relevant.is_empty():
        return {}
    wide = relevant.pivot(values="close", index="date", on="asset_id").sort("date")
    asset_cols = [c for c in wide.columns if c != "date"]
    if not asset_cols:
        return {}

    n = wide.height
    max_lb = max(cfg.lookbacks_days)
    if n <= max_lb:
        return {}

    arr = wide.select(asset_cols).to_numpy()  # (n, n_assets)
    last = arr[-1]  # (n_assets,)

    rocs: list[np.ndarray] = []
    for lb in cfg.lookbacks_days:
        prior = arr[-1 - lb]
        with np.errstate(divide="ignore", invalid="ignore"):
            roc = np.where(prior > 0, last / prior - 1.0, np.nan)
        rocs.append(roc)
    score_arr = np.mean(np.stack(rocs), axis=0)

    out: dict[str, float] = {}
    for i, asset_id in enumerate(asset_cols):
        v = score_arr[i]
        if not np.isnan(v):
            out[asset_id] = float(v)
    return out


def _score_series(closes: list[float], lookbacks: tuple[int, ...]) -> float | None:
    """Single-asset score computation. Kept as a building block for tests and
    callers operating on a raw close series."""
    if not closes:
        return None
    last = closes[-1]
    rocs: list[float] = []
    for lb in lookbacks:
        if len(closes) <= lb:
            return None
        prior = closes[-1 - lb]
        if prior <= 0:
            return None
        rocs.append(last / prior - 1.0)
    return statistics.fmean(rocs)
