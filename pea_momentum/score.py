"""Accelerated dual-momentum scoring.

For each asset, score(t) = aggregate over lookbacks of (close(t) / close(t - L) - 1).
Computed on EUR-denominated ETF closes (or the €STR-derived synthetic for the
safe asset). Output is a vector of scores indexed by asset_id, evaluated at a
single signal date.

Three aggregations are supported across the configured lookbacks:
- ``mean``    arithmetic average — the default ADM aggregation.
- ``median``  trimmed-tails view — robust to a single anomalous lookback.
- ``min``     pessimistic — requires every lookback to be positive for the
              score to clear the absolute filter; useful as a sensitivity exhibit.
"""

from __future__ import annotations

import statistics
from datetime import date

import polars as pl

from .universe import Scoring

AGGREGATION_MEAN = "mean"
AGGREGATION_MEDIAN = "median"
AGGREGATION_MIN = "min"
SUPPORTED_AGGREGATIONS: frozenset[str] = frozenset(
    {AGGREGATION_MEAN, AGGREGATION_MEDIAN, AGGREGATION_MIN}
)


def score_at(
    prices_long: pl.DataFrame,
    asset_ids: list[str],
    as_of: date,
    cfg: Scoring,
) -> dict[str, float]:
    """Compute scores for `asset_ids` at `as_of` using `cfg.lookbacks_days`.

    Missing assets or insufficient history yield no entry in the output. The
    backtest treats missing scores as "ineligible this period".

    Pure-polars: per-asset shifts compute every lookback's ROC in a single
    expression sweep, then horizontal aggregation collapses them to one
    score per asset at the most-recent close ≤ as_of.
    """
    if cfg.aggregation not in SUPPORTED_AGGREGATIONS:
        raise ValueError(
            f"Unsupported aggregation: {cfg.aggregation!r} "
            f"(supported: {sorted(SUPPORTED_AGGREGATIONS)})"
        )

    relevant = prices_long.filter(pl.col("asset_id").is_in(asset_ids) & (pl.col("date") <= as_of))
    if relevant.is_empty():
        return {}

    sorted_long = relevant.sort(["asset_id", "date"])
    roc_cols = [f"_roc_{lb}" for lb in cfg.lookbacks_days]
    # Per-asset shift: close at row N divided by close at row N-L gives the
    # L-day ROC at every row. We only need the last row per asset; the
    # subsequent filter selects it.
    with_rocs = sorted_long.with_columns(
        *[
            (
                pl.col("close")
                / pl.when(pl.col("close").shift(lb).over("asset_id") > 0)
                .then(pl.col("close").shift(lb).over("asset_id"))
                .otherwise(None)
                - 1.0
            ).alias(f"_roc_{lb}")
            for lb in cfg.lookbacks_days
        ]
    )

    last_per_asset = with_rocs.filter(pl.col("date") == pl.col("date").max().over("asset_id"))

    if cfg.aggregation == AGGREGATION_MEAN:
        score_expr = pl.mean_horizontal(roc_cols)
    elif cfg.aggregation == AGGREGATION_MIN:
        score_expr = pl.min_horizontal(roc_cols)
    else:  # AGGREGATION_MEDIAN — polars has no median_horizontal, so list-then-median.
        score_expr = pl.concat_list(roc_cols).list.median()

    # `*_horizontal` skip-nulls by default; we want strict propagation so that
    # an asset with insufficient history (any lookback null) is excluded
    # rather than silently scored on a partial average.
    all_present = pl.all_horizontal([pl.col(c).is_not_null() for c in roc_cols])
    scored = last_per_asset.with_columns(
        _score=pl.when(all_present).then(score_expr).otherwise(None)
    ).drop_nulls(subset=["_score"])
    return dict(
        zip(
            scored.get_column("asset_id").to_list(),
            scored.get_column("_score").to_list(),
            strict=True,
        )
    )


def _score_series(
    closes: list[float],
    lookbacks: tuple[int, ...],
    aggregation: str = AGGREGATION_MEAN,
) -> float | None:
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
    if aggregation == AGGREGATION_MEAN:
        return statistics.fmean(rocs)
    if aggregation == AGGREGATION_MEDIAN:
        return statistics.median(rocs)
    if aggregation == AGGREGATION_MIN:
        return min(rocs)
    raise ValueError(f"Unsupported aggregation: {aggregation!r}")
