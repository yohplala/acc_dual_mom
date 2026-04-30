"""Price fetching: yfinance for ETFs, ECB Data Portal for €STR.

Boundary module — yfinance returns pandas, we convert to polars on ingress.
"""

from __future__ import annotations

import io
import logging
from datetime import date, datetime, timedelta

import httpx
import polars as pl
import yfinance as yf

from .universe import Asset, Config, SafeAsset

log = logging.getLogger(__name__)

ECB_ESTR_URL = "https://data-api.ecb.europa.eu/service/data/EST/B.EU000A2X2A25.WT?format=csvdata"
"""€STR volume-weighted trimmed-mean rate (annualized, ACT/360, business days)."""

ESTR_BASE_PRICE = 100.0
"""Synthetic price level on the first available €STR fixing."""


def fetch_all(config: Config, start: date | None = None) -> pl.DataFrame:
    start = start or _earliest_useful_start()
    frames: list[pl.DataFrame] = [fetch_yahoo(asset, start=start) for asset in config.assets]
    frames.append(fetch_safe_asset(config.safe_asset, start=start))
    return pl.concat([f for f in frames if not f.is_empty()])


def fetch_yahoo(asset: Asset, start: date) -> pl.DataFrame:
    log.info("fetching %s (%s) from yfinance since %s", asset.id, asset.yahoo, start)
    raw = yf.download(
        asset.yahoo,
        start=start.isoformat(),
        progress=False,
        auto_adjust=True,
        actions=False,
    )
    if raw is None or raw.empty:
        raise FetchError(f"yfinance returned no data for {asset.yahoo} ({asset.id})")

    if hasattr(raw.columns, "get_level_values"):
        raw = raw.droplevel(1, axis=1) if raw.columns.nlevels > 1 else raw

    if "Close" not in raw.columns:
        raise FetchError(f"yfinance response for {asset.yahoo} missing Close column")

    # pandas 3 may return Float64 (nullable extension) which requires pyarrow
    # for `pl.from_pandas`. Build polars from native numpy arrays instead.
    closes = raw["Close"].to_numpy(dtype="float64", na_value=float("nan"))
    dates = raw.index.to_numpy()  # datetime64; polars casts cleanly to Date

    return (
        pl.DataFrame({"date": dates, "close": closes})
        .with_columns(
            pl.col("date").cast(pl.Date),
            pl.lit(asset.id).alias("asset_id"),
            pl.lit("yfinance").alias("source"),
        )
        .filter(pl.col("close").is_not_null() & pl.col("close").is_not_nan())
        .select(["date", "asset_id", "close", "source"])
    )


def fetch_safe_asset(safe: SafeAsset, start: date) -> pl.DataFrame:
    if safe.proxy != "estr":
        raise FetchError(f"Unsupported safe-asset proxy: {safe.proxy}")
    estr = fetch_estr(start=start)
    if estr.is_empty():
        raise FetchError("€STR fetch returned no data")
    return _estr_to_synthetic_close(estr, asset_id=safe.id)


def fetch_estr(start: date) -> pl.DataFrame:
    """Fetch €STR daily fixings from ECB Data Portal, return [date, rate_pct]."""
    log.info("fetching €STR from ECB since %s", start)
    with httpx.Client(timeout=30.0) as client:
        r = client.get(ECB_ESTR_URL, headers={"Accept": "text/csv"})
        r.raise_for_status()

    raw = pl.read_csv(io.BytesIO(r.content))
    date_col = next((c for c in raw.columns if c.upper() == "TIME_PERIOD"), None)
    value_col = next((c for c in raw.columns if c.upper() == "OBS_VALUE"), None)
    if date_col is None or value_col is None:
        raise FetchError(
            f"unexpected ECB CSV columns: {raw.columns}; expected TIME_PERIOD + OBS_VALUE"
        )

    return (
        raw.select(
            pl.col(date_col).str.strptime(pl.Date, format="%Y-%m-%d").alias("date"),
            pl.col(value_col).cast(pl.Float64).alias("rate_pct"),
        )
        .filter(pl.col("date") >= start)
        .sort("date")
    )


def _estr_to_synthetic_close(estr: pl.DataFrame, asset_id: str) -> pl.DataFrame:
    """Compound €STR daily fixings into a synthetic price series.

    Convention: rate is annualized ACT/360. One business day's accrual factor
    is (1 + rate / 100 / 360). The published fixing for date D applies from D
    overnight to the next business day. We compound on each published fixing
    so the resulting series has values only on business days; consumers
    forward-fill if needed.
    """
    return (
        estr.sort("date")
        .with_columns(
            daily_factor=(1.0 + pl.col("rate_pct") / 100.0 / 360.0),
        )
        .with_columns(
            cum_factor=pl.col("daily_factor").cum_prod(),
        )
        .with_columns(
            close=pl.col("cum_factor") * ESTR_BASE_PRICE,
            asset_id=pl.lit(asset_id),
            source=pl.lit("estr_synthetic"),
        )
        .select(["date", "asset_id", "close", "source"])
    )


def _earliest_useful_start() -> date:
    """Default start: ~12 years back, captures all UCITS ETF inceptions in our list."""
    return (datetime.utcnow().date() - timedelta(days=365 * 12)).replace(month=1, day=1)


class FetchError(RuntimeError):
    """Raised when an upstream price provider returns unusable data."""
