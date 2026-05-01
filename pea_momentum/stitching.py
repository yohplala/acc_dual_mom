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
"""

from __future__ import annotations

from datetime import date

import polars as pl


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
    """
    if etf_long.is_empty():
        return etf_long
    if proxy_long.is_empty():
        return etf_long

    etf_at = etf_long.filter(pl.col("date") >= inception).sort("date").head(1)
    proxy_at = proxy_long.filter(pl.col("date") <= inception).sort("date").tail(1)
    if etf_at.is_empty() or proxy_at.is_empty():
        return etf_long

    etf_close = float(etf_at.get_column("close")[0])
    proxy_close = float(proxy_at.get_column("close")[0])
    if proxy_close <= 0:
        return etf_long
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
    """
    return _convert(idx_usd, eurusd)


def jpy_to_eur(idx_jpy: pl.DataFrame, eurjpy: pl.DataFrame) -> pl.DataFrame:
    """Convert a JPY-denominated `[date, close]` series to EUR using a
    daily `[date, close]` EURJPY series (JPY per EUR). EUR price = JPY / EURJPY.
    """
    return _convert(idx_jpy, eurjpy)


def _convert(idx_local: pl.DataFrame, fx_per_eur: pl.DataFrame) -> pl.DataFrame:
    """Local-CCY index series → EUR. fx_per_eur is units-of-local-ccy per 1 EUR."""
    if idx_local.is_empty() or fx_per_eur.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "close": pl.Float64})
    fx = fx_per_eur.select(["date", pl.col("close").alias("_fx")])
    joined = idx_local.join(fx, on="date", how="inner")
    joined = joined.filter(pl.col("_fx") > 0)
    return joined.with_columns(close=pl.col("close") / pl.col("_fx")).select(["date", "close"])
