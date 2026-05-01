"""Price fetching: yfinance for ETFs, ECB Data Portal for €STR.

Boundary module — yfinance returns pandas, we convert to polars on ingress.
"""

from __future__ import annotations

import io
import logging
from datetime import date

import httpx
import polars as pl
import yfinance as yf

from . import stitching
from .universe import Asset, Config, SafeAsset

log = logging.getLogger(__name__)

ECB_ESTR_URL = "https://data-api.ecb.europa.eu/service/data/EST/B.EU000A2X2A25.WT?format=csvdata"
"""€STR volume-weighted trimmed-mean rate (annualized, ACT/360, business days)."""

ECB_EONIA_URL = (
    "https://data-api.ecb.europa.eu/service/data/FM/D.U2.EUR.4F.MM.EONIA.HSTA?format=csvdata"
)
"""EONIA daily rate — predecessor to €STR. Available from 1999-01-04 to 2022-01-03,
overlaps with €STR from 2019-10-02. We use EONIA pre-2019-10-02 and €STR after."""

ESTR_BASE_PRICE = 100.0
"""Synthetic price level on the first available safe-asset fixing."""

ESTR_START = date(2019, 10, 2)
"""First €STR fixing date. Pre-this we splice EONIA."""

YAHOO_FX = {
    "usd_pr": "EURUSD=X",
    "jpy_pr": "EURJPY=X",
    "eur_pr": None,  # no FX needed
}


def fetch_all(config: Config, start: date | None = None) -> pl.DataFrame:
    start = start or _earliest_useful_start()
    frames: list[pl.DataFrame] = [
        fetch_yahoo_with_optional_proxy(asset, start=start) for asset in config.assets
    ]
    frames.append(fetch_safe_asset(config.safe_asset, start=start))
    return pl.concat([f for f in frames if not f.is_empty()])


def fetch_yahoo_with_optional_proxy(asset: Asset, start: date) -> pl.DataFrame:
    """Fetch the live ETF, then splice pre-inception index proxy if configured."""
    etf = fetch_yahoo(asset, start=start)
    if asset.index_proxy is None or asset.inception is None:
        return etf
    try:
        proxy = _fetch_proxy_in_eur(asset, start=start)
    except FetchError as exc:
        log.warning("proxy fetch failed for %s — keeping ETF-only history: %s", asset.id, exc)
        return etf
    return stitching.splice_at_inception(etf, proxy, asset.inception, asset.id)


def _fetch_proxy_in_eur(asset: Asset, start: date) -> pl.DataFrame:
    """Fetch the proxy index and convert to EUR if needed. Returns [date, close]."""
    kind = asset.index_proxy_kind or "eur_pr"
    if kind not in YAHOO_FX:
        raise FetchError(f"Unknown index_proxy_kind {kind!r} for asset {asset.id}")
    if asset.index_proxy is None:
        raise FetchError(f"Asset {asset.id} has no index_proxy configured")

    if asset.index_proxy_source == "stooq":
        idx = _fetch_stooq_close_only(asset.index_proxy, start=start)
    elif asset.index_proxy_source == "yahoo":
        idx = _fetch_yahoo_close_only(asset.index_proxy, start=start)
    else:
        raise FetchError(
            f"Unknown index_proxy_source {asset.index_proxy_source!r} for asset {asset.id}"
        )

    fx_ticker = YAHOO_FX[kind]
    if fx_ticker is None:
        return idx

    fx = _fetch_yahoo_close_only(fx_ticker, start=start)
    if kind == "usd_pr":
        return stitching.usd_to_eur(idx, fx)
    if kind == "jpy_pr":
        return stitching.jpy_to_eur(idx, fx)
    raise FetchError(f"FX conversion not implemented for kind {kind!r}")


def _fetch_yahoo_close_only(ticker: str, start: date) -> pl.DataFrame:
    """Fetch a Yahoo ticker and return [date, close] only (no asset_id/source)."""
    log.info("fetching proxy/FX %s from yfinance since %s", ticker, start)
    raw = yf.download(
        ticker, start=start.isoformat(), progress=False, auto_adjust=True, actions=False
    )
    if raw is None or raw.empty:
        raise FetchError(f"yfinance returned no data for {ticker}")
    if hasattr(raw.columns, "get_level_values"):
        raw = raw.droplevel(1, axis=1) if raw.columns.nlevels > 1 else raw
    if "Close" not in raw.columns:
        raise FetchError(f"yfinance response for {ticker} missing Close column")
    closes = raw["Close"].to_numpy(dtype="float64", na_value=float("nan"))
    dates = raw.index.to_numpy().astype("datetime64[ns]")
    return (
        pl.DataFrame({"date": dates, "close": closes})
        .with_columns(pl.col("date").cast(pl.Date))
        .filter(pl.col("close").is_not_null() & pl.col("close").is_not_nan())
        .sort("date")
    )


STOOQ_BASE_URL = "https://stooq.com/q/d/l/"
"""Stooq's daily-CSV endpoint. Pattern:
    https://stooq.com/q/d/l/?s=<ticker>&d1=<YYYYMMDD>&d2=<YYYYMMDD>&i=d
Returns CSV with header `Date,Open,High,Low,Close,Volume`.
"""


def _fetch_stooq_close_only(ticker: str, start: date) -> pl.DataFrame:
    """Fetch a Stooq ticker via CSV download and return [date, close].

    Stooq has TR / Reinvested variants for many European indices that Yahoo
    lacks (^stoxxr, ^sx5gr, etc). The CSV endpoint accepts a ticker and date
    range — we pull the full range and filter to `start` afterwards.
    """
    end = date.today()
    params = {
        "s": ticker,
        "d1": start.strftime("%Y%m%d"),
        "d2": end.strftime("%Y%m%d"),
        "i": "d",
    }
    log.info("fetching proxy %s from Stooq since %s", ticker, start)
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        r = client.get(STOOQ_BASE_URL, params=params, headers={"Accept": "text/csv"})
        r.raise_for_status()

    body = r.content
    # Stooq sometimes returns an HTML error page on bad tickers; CSV starts with
    # "Date," — bail loudly on anything else.
    if not body.lstrip().lower().startswith(b"date,"):
        raise FetchError(
            f"Stooq returned non-CSV response for {ticker!r} " f"(first 80 bytes: {body[:80]!r})"
        )

    raw = pl.read_csv(io.BytesIO(body))
    if "Date" not in raw.columns or "Close" not in raw.columns:
        raise FetchError(f"unexpected Stooq CSV columns for {ticker}: {raw.columns}")

    return (
        raw.select(
            pl.col("Date").str.strptime(pl.Date, format="%Y-%m-%d").alias("date"),
            pl.col("Close").cast(pl.Float64).alias("close"),
        )
        .filter(pl.col("close").is_not_null() & pl.col("close").is_not_nan())
        .filter(pl.col("date") >= start)
        .sort("date")
    )


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
    # Polars only accepts datetime64 at D/ms/us/ns resolution — yfinance under
    # pandas 3 yields seconds-resolution indices, so we normalize to ns here.
    closes = raw["Close"].to_numpy(dtype="float64", na_value=float("nan"))
    dates = raw.index.to_numpy().astype("datetime64[ns]")

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
    """Fetch the safe-asset synthetic price series.

    €STR is only available from 2019-10-02 onwards. For backtests starting
    before that date we splice EONIA (1999-2022) onto the front of €STR so
    the safe asset has continuous history back to 1999.
    """
    if safe.proxy != "estr":
        raise FetchError(f"Unsupported safe-asset proxy: {safe.proxy}")

    estr = fetch_estr(start=max(start, ESTR_START))
    if estr.is_empty():
        raise FetchError("€STR fetch returned no data")

    if start >= ESTR_START:
        return _rates_to_synthetic_close(estr, asset_id=safe.id)

    # Need pre-2019 history → fetch EONIA, splice on the day before €STR starts
    try:
        eonia = fetch_eonia(start=start)
    except FetchError as exc:
        log.warning("EONIA fetch failed; safe asset limited to €STR era: %s", exc)
        return _rates_to_synthetic_close(estr, asset_id=safe.id)
    if eonia.is_empty():
        return _rates_to_synthetic_close(estr, asset_id=safe.id)

    # Concatenate: EONIA before ESTR_START, €STR from ESTR_START. Both are
    # daily rates in % ann ACT/360. Compound the combined series into a price.
    eonia_pre = eonia.filter(pl.col("date") < ESTR_START)
    combined = pl.concat([eonia_pre, estr]).sort("date").unique(subset=["date"], keep="last")
    return _rates_to_synthetic_close(combined, asset_id=safe.id)


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


def fetch_eonia(start: date) -> pl.DataFrame:
    """Fetch EONIA daily fixings from ECB Data Portal, return [date, rate_pct].

    EONIA was the eurozone overnight rate from 1999-01-04 until it was
    discontinued on 2022-01-03 (replaced by €STR which started 2019-10-02).
    For backtests starting before €STR's launch, EONIA fills in the pre-2019
    history. Same ACT/360 convention as €STR.
    """
    log.info("fetching EONIA from ECB since %s", start)
    with httpx.Client(timeout=30.0) as client:
        r = client.get(ECB_EONIA_URL, headers={"Accept": "text/csv"})
        r.raise_for_status()

    raw = pl.read_csv(io.BytesIO(r.content))
    date_col = next((c for c in raw.columns if c.upper() == "TIME_PERIOD"), None)
    value_col = next((c for c in raw.columns if c.upper() == "OBS_VALUE"), None)
    if date_col is None or value_col is None:
        raise FetchError(
            f"unexpected ECB EONIA CSV columns: {raw.columns}; expected TIME_PERIOD + OBS_VALUE"
        )

    return (
        raw.select(
            pl.col(date_col).str.strptime(pl.Date, format="%Y-%m-%d").alias("date"),
            pl.col(value_col).cast(pl.Float64).alias("rate_pct"),
        )
        .filter(pl.col("date") >= start)
        .filter(pl.col("rate_pct").is_not_null())
        .sort("date")
    )


def _rates_to_synthetic_close(rates: pl.DataFrame, asset_id: str) -> pl.DataFrame:
    """Compound daily fixings into a synthetic price series.

    Convention: rate is annualized ACT/360. One business day's accrual factor
    is (1 + rate / 100 / 360). The published fixing for date D applies from D
    overnight to the next business day. We compound on each published fixing
    so the resulting series has values only on business days; consumers
    forward-fill if needed.
    """
    return (
        rates.sort("date")
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
    """Default start: 2008-01-01, far enough to include the 2008 GFC for stitched
    strategies. Assets without an `index_proxy` configured will simply have
    no data before their ETF inception — backtests filter to the period when
    all required assets are present."""
    return date(2008, 1, 1)


class FetchError(RuntimeError):
    """Raised when an upstream price provider returns unusable data."""
