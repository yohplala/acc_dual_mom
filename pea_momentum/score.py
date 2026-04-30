"""Accelerated dual-momentum scoring.

For each asset, score(t) = mean over lookbacks of (close(t) / close(t - L) - 1).
Computed on EUR-denominated ETF closes (or the €STR-derived synthetic for the
safe asset). Output is a vector of scores indexed by asset_id, evaluated at a
single signal date.
"""

from __future__ import annotations

import statistics
from datetime import date

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
    """
    if cfg.aggregation != "mean":
        raise ValueError(f"Only aggregation=mean is supported (got {cfg.aggregation!r})")

    relevant = prices_long.filter(pl.col("asset_id").is_in(asset_ids) & (pl.col("date") <= as_of))
    out: dict[str, float] = {}
    for asset_id in asset_ids:
        series = (
            relevant.filter(pl.col("asset_id") == asset_id)
            .sort("date")
            .get_column("close")
            .to_list()
        )
        score = _score_series(series, cfg.lookbacks_days)
        if score is not None:
            out[asset_id] = score
    return out


def _score_series(closes: list[float], lookbacks: tuple[int, ...]) -> float | None:
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
