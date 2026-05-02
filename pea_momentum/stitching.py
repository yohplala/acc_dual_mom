"""Splice an index-proxy series onto a live ETF series, return-chained.

Most PEA-eligible UCITS ETFs in our universe launched between 2014 and 2024,
so a 2008+ backtest needs pre-launch history from the underlying index. The
splice approach:

1. Fetch the ETF series (post-inception only)
2. Fetch the index proxy series in the proxy's own currency
3. Convert the proxy to EUR via daily FX rates if needed
4. Rescale the proxy levels so that the proxy's level on the inception date
   matches the ETF's level on the same date — pre-inception levels are
   walked back from there using the proxy's own returns
5. Concatenate: scaled proxy for date < inception, ETF for date >= inception

Resulting series is continuous in level (no jump on the splice date) and
provenance is preserved via the `source` column ("yfinance" for live ETF,
"stitched_index_proxy" for synthetic pre-inception segment).

Failures in any of these steps raise `FetchError` — the caller decides
whether to engage `index_proxy_fallback` for graceful degradation.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from .errors import FetchError

# Hard ceiling on plausible single-day moves for any equity index or ETF.
# Worst real single-day move on a major equity index since 1928 is Black
# Monday 1987 at -22.6%. COVID 2020 worst day was -12%. Anything beyond
# this threshold is bad data — yfinance has been observed to return
# half-priced or doubled values on isolated dates for both FX series
# (EURUSD=X) and `^XXX` index tickers (^GSPC, ^STOXX50E). Failing loud
# beats silently producing impossible spikes in the equity curve.
MAX_PLAUSIBLE_DAILY_RETURN = 0.30


def _validate_returns_or_raise(
    series: pl.DataFrame, asset_id: str, label: str, threshold: float = MAX_PLAUSIBLE_DAILY_RETURN
) -> None:
    """Raise `FetchError` if any single-day return on `series` exceeds the
    plausibility ceiling. `series` must have `[date, close]` sorted ascending.
    `label` identifies the segment for the error message ("proxy", "live ETF",
    or "spliced")."""
    if series.height < 2:
        return
    rets = series.sort("date").with_columns(ret=(pl.col("close") / pl.col("close").shift(1) - 1.0))
    bad = rets.filter(pl.col("ret").abs() > threshold)
    if bad.is_empty():
        return
    sample = bad.select(["date", "close", "ret"]).head(5).rows()
    raise FetchError(
        f"splice {asset_id}: {label} series has {bad.height} day(s) with "
        f"|return| > {threshold:.0%} — implausible for any equity index, "
        f"strongly suggests bad upstream data (yfinance is known to return "
        f"half-priced/doubled values for `^XXX` indices and EURUSD=X on "
        f"isolated dates). First 5: "
        + "; ".join(f"{d} close={c:.4f} ret={r:+.3f}" for d, c, r in sample)
    )


def splice_at_inception(
    etf_long: pl.DataFrame,
    proxy_long: pl.DataFrame,
    inception: date,
    asset_id: str,
) -> pl.DataFrame:
    """Splice pre-inception proxy onto post-inception ETF.

    Both inputs are long-format `[date, close]` DataFrames (proxy may also
    carry extra columns; only `date` and `close` are used). Returns a long
    `[date, asset_id, close, source]` covering the union of dates.

    Raises `FetchError` if any precondition fails (empty inputs, no overlap
    around the inception date, non-positive proxy close, or any single-day
    return on either segment exceeds 30% — the plausibility ceiling for an
    equity index. A user-configured proxy that can't be spliced is a real
    problem — not silently masked.
    """
    if etf_long.is_empty():
        raise FetchError(
            f"splice {asset_id}: ETF series is empty (yfinance returned no rows for the live ETF)"
        )
    if proxy_long.is_empty():
        raise FetchError(f"splice {asset_id}: proxy series is empty (check the index_proxy ticker)")

    # Validate before scaling so the error message points at the raw upstream
    # series. The proxy-side check catches yfinance bad-day artefacts; the
    # ETF-side check catches the same on the live product (rarer but possible).
    _validate_returns_or_raise(proxy_long, asset_id, "proxy")
    _validate_returns_or_raise(etf_long, asset_id, "live ETF")

    etf_at = etf_long.filter(pl.col("date") >= inception).sort("date").head(1)
    proxy_at = proxy_long.filter(pl.col("date") <= inception).sort("date").tail(1)
    if etf_at.is_empty():
        raise FetchError(f"splice {asset_id}: no ETF data on or after inception date {inception}")
    if proxy_at.is_empty():
        raise FetchError(
            f"splice {asset_id}: no proxy data on or before inception date {inception} "
            f"(proxy series begins after the configured inception)"
        )

    etf_close = float(etf_at.get_column("close")[0])
    proxy_close = float(proxy_at.get_column("close")[0])
    if proxy_close <= 0:
        raise FetchError(
            f"splice {asset_id}: proxy close at inception is non-positive ({proxy_close})"
        )
    scale = etf_close / proxy_close

    pre = (
        proxy_long.filter(pl.col("date") < inception)
        .sort("date")
        .with_columns(
            close=pl.col("close") * scale,
            asset_id=pl.lit(asset_id),
            source=pl.lit("stitched_index_proxy"),
        )
        .select(["date", "asset_id", "close", "source"])
    )

    post = etf_long.select(["date", "asset_id", "close", "source"])
    return pl.concat([pre, post]).sort("date")


def usd_to_eur(idx_usd: pl.DataFrame, eurusd: pl.DataFrame) -> pl.DataFrame:
    """Convert a USD-denominated `[date, close]` series to EUR using a
    daily `[date, close]` EURUSD series (USD per EUR). EUR price = USD / EURUSD.
    Raises `FetchError` if either input is empty.
    """
    return _convert(idx_usd, eurusd, ccy="USD")


def jpy_to_eur(idx_jpy: pl.DataFrame, eurjpy: pl.DataFrame) -> pl.DataFrame:
    """Convert a JPY-denominated `[date, close]` series to EUR using a
    daily `[date, close]` EURJPY series (JPY per EUR). EUR price = JPY / EURJPY.
    Raises `FetchError` if either input is empty.
    """
    return _convert(idx_jpy, eurjpy, ccy="JPY")


def _convert(idx_local: pl.DataFrame, fx_per_eur: pl.DataFrame, ccy: str) -> pl.DataFrame:
    """Local-CCY index series → EUR. fx_per_eur is units-of-local-ccy per 1 EUR.
    Raises `FetchError` rather than silently returning empty when an input
    is empty or the inner-join produces no overlapping dates.

    The FX series is independently sanity-checked for impossible single-day
    moves: a bad day on EURUSD=X (yfinance has been observed to return
    half/double values on isolated dates) silently corrupts every cross-
    currency proxy by exactly the same factor. Catching it at the FX layer
    is more informative than detecting the resulting equity-curve spike
    downstream.
    """
    if idx_local.is_empty():
        raise FetchError(f"{ccy}→EUR: index series is empty")
    if fx_per_eur.is_empty():
        raise FetchError(f"{ccy}→EUR: FX series is empty (check EUR{ccy}=X yfinance ticker)")

    # Validate the FX series itself — a 30% one-day move in a major-pair
    # spot rate is implausible (worst-ever EURUSD daily move ≈ 4-5%).
    _validate_returns_or_raise(fx_per_eur, asset_id=f"EUR{ccy}=X", label="FX")

    fx = fx_per_eur.select(["date", pl.col("close").alias("_fx")])
    joined = idx_local.join(fx, on="date", how="inner")
    joined = joined.filter(
        (pl.col("_fx") > 0) & pl.col("_fx").is_not_nan() & pl.col("close").is_not_nan()
    )
    if joined.is_empty():
        raise FetchError(
            f"{ccy}→EUR: no overlapping dates between index and FX series after positivity filter"
        )
    return joined.with_columns(close=pl.col("close") / pl.col("_fx")).select(["date", "close"])
