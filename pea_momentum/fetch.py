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
from .errors import FetchError
from .universe import Asset, Config

log = logging.getLogger(__name__)

ECB_ESTR_URL = "https://data-api.ecb.europa.eu/service/data/EST/B.EU000A2X2A25.WT?format=csvdata"
"""€STR volume-weighted trimmed-mean rate (annualized, ACT/360, business days)."""

ECB_EONIA_URL = "https://data-api.ecb.europa.eu/service/data/EON/D.EONIA_TO.RATE?format=csvdata"
"""EONIA daily rate — predecessor to €STR. Series key `EON.D.EONIA_TO.RATE`
(dataflow EON, daily, total/aggregate, rate). Available from 1999-01-04 to
2022-01-03, overlaps with €STR from 2019-10-02. We use EONIA pre-2019-10-02
and €STR after."""

ESTR_BASE_PRICE = 100.0
"""Synthetic price level on the first available safe-asset fixing."""

ESTR_START = date(2019, 10, 2)
"""First €STR fixing date. Pre-this we splice EONIA."""

# Supported `index_proxy_kind` values. The label encodes only the FX-conversion
# path; whether the underlying series is PR or TR depends on the chosen ticker.
PROXY_KIND_EUR_TR = "eur_tr"
PROXY_KIND_USD_TR = "usd_tr"
SUPPORTED_PROXY_KINDS: frozenset[str] = frozenset({PROXY_KIND_EUR_TR, PROXY_KIND_USD_TR})


def fetch_all(config: Config, start: date | None = None) -> pl.DataFrame:
    """Fetch every active asset's price series. Assets with `synth_proxy` set
    are synthesised (currently only `synth_proxy=estr` is supported, compounded
    from ECB €STR fixings); other assets pull yfinance with optional pre-
    inception index-proxy stitching."""
    start = start or _earliest_useful_start()
    frames: list[pl.DataFrame] = []
    for asset in config.assets:
        if asset.synth_proxy is not None:
            frames.append(fetch_synth_asset(asset, start=start))
        else:
            frames.append(fetch_yahoo_with_optional_proxy(asset, start=start))
    return pl.concat([f for f in frames if not f.is_empty()])


def fetch_yahoo_with_optional_proxy(asset: Asset, start: date) -> pl.DataFrame:
    """Fetch the live ETF, then splice pre-inception index proxy if configured.

    Proxy failures propagate as `FetchError` (loud failure) — a configured proxy
    that can't be fetched should be fixed in YAML, not masked by ETF-only
    history.
    """
    etf = fetch_yahoo(asset, start=start)
    if asset.index_proxy is None or asset.inception is None:
        return etf
    proxy = _fetch_proxy_in_eur(asset, start=start)
    return stitching.splice_at_inception(etf, proxy, asset.inception, asset.id)


def _fetch_proxy_in_eur(asset: Asset, start: date) -> pl.DataFrame:
    """Fetch the proxy index from Yahoo and convert to EUR if needed.

    Returns `[date, close]`. Raises `FetchError` if the proxy ticker is missing
    or the kind is unsupported.
    """
    if asset.index_proxy is None:
        raise FetchError(f"Asset {asset.id} has no index_proxy configured")
    kind = asset.index_proxy_kind or PROXY_KIND_EUR_TR
    if kind not in SUPPORTED_PROXY_KINDS:
        raise FetchError(
            f"Unsupported index_proxy_kind {kind!r} for asset {asset.id} "
            f"(supported: {sorted(SUPPORTED_PROXY_KINDS)})"
        )

    idx = _fetch_yahoo_close_only(asset.index_proxy, start=start)
    if kind == PROXY_KIND_EUR_TR:
        return idx
    fx = _fetch_yahoo_close_only("EURUSD=X", start=start)
    return stitching.usd_to_eur(idx, fx)


def _fetch_yahoo_close_only(ticker: str, start: date) -> pl.DataFrame:
    """Fetch a Yahoo ticker and return [date, close] only (no asset_id/source)."""
    log.info("fetching %s from yfinance since %s", ticker, start)
    raw = yf.download(
        ticker, start=start.isoformat(), progress=False, auto_adjust=True, actions=False
    )
    if raw is None or raw.empty:
        raise FetchError(f"yfinance returned no data for {ticker}")
    if raw.columns.nlevels > 1:
        raw = raw.droplevel(1, axis=1)
    if "Close" not in raw.columns:
        raise FetchError(f"yfinance response for {ticker} missing Close column")
    # pandas 3 may return Float64 (nullable extension) which requires pyarrow
    # for `pl.from_pandas`. Build polars from native numpy arrays instead.
    # Polars only accepts datetime64 at D/ms/us/ns resolution — yfinance under
    # pandas 3 yields seconds-resolution indices, so we normalize to ns here.
    closes = raw["Close"].to_numpy(dtype="float64", na_value=float("nan"))
    dates = raw.index.to_numpy().astype("datetime64[ns]")
    return (
        pl.DataFrame({"date": dates, "close": closes})
        .with_columns(pl.col("date").cast(pl.Date))
        .filter(pl.col("close").is_not_null() & pl.col("close").is_not_nan())
        .sort("date")
    )


def fetch_yahoo(asset: Asset, start: date) -> pl.DataFrame:
    """Fetch the live ETF closes via Yahoo. Returns [date, asset_id, close, source]."""
    return (
        _fetch_yahoo_close_only(asset.yahoo, start=start)
        .with_columns(
            pl.lit(asset.id).alias("asset_id"),
            pl.lit("yfinance").alias("source"),
        )
        .select(["date", "asset_id", "close", "source"])
    )


def fetch_synth_asset(asset: Asset, start: date) -> pl.DataFrame:
    """Fetch a synthetic price series for an asset whose `synth_proxy` field
    is set. Currently only `synth_proxy=estr` is supported: compounds ECB
    €STR fixings (with EONIA splice for pre-2019 history) into a price.

    €STR is only available from 2019-10-02 onwards. For backtests starting
    before that date we splice EONIA (1999-2022) onto the front of €STR so
    the synthetic series has continuous history back to 1999.
    """
    if asset.synth_proxy != "estr":
        raise FetchError(f"Unsupported synth_proxy={asset.synth_proxy!r} for asset {asset.id}")

    estr = fetch_estr(start=max(start, ESTR_START))
    if estr.is_empty():
        raise FetchError("€STR fetch returned no data")

    if start >= ESTR_START:
        return _rates_to_synthetic_close(estr, asset_id=asset.id)

    # Need pre-2019 history → fetch EONIA, splice on the day before €STR starts.
    # Failures propagate (loud failure): silent fallback to €STR-only would
    # silently truncate the synthetic series to 2019+. Better to crash so
    # the user learns the EONIA endpoint is broken and can fix it.
    eonia = fetch_eonia(start=start)
    if eonia.is_empty():
        raise FetchError(
            f"EONIA fetch returned empty data — cannot extend {asset.id} pre-"
            f"{ESTR_START}. Check ECB Data Portal series `EON.D.EONIA_TO.RATE`."
        )

    # Concatenate: EONIA before ESTR_START, €STR from ESTR_START. Both are
    # daily rates in % ann ACT/360. Compound the combined series into a price.
    eonia_pre = eonia.filter(pl.col("date") < ESTR_START)
    combined = pl.concat([eonia_pre, estr]).sort("date").unique(subset=["date"], keep="last")
    return _rates_to_synthetic_close(combined, asset_id=asset.id)


def _fetch_ecb_csv(url: str, start: date, *, name: str) -> pl.DataFrame:
    """Fetch a daily ECB rate CSV and return [date, rate_pct].

    All HTTP / parsing errors are converted to `FetchError` so callers'
    fallback paths engage instead of crashing the whole fetch.
    """
    log.info("fetching %s from ECB since %s", name, start)
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.get(url, headers={"Accept": "text/csv"})
            r.raise_for_status()
    except httpx.HTTPError as exc:
        raise FetchError(f"{name} fetch failed: {exc}") from exc

    raw = pl.read_csv(io.BytesIO(r.content))
    date_col = next((c for c in raw.columns if c.upper() == "TIME_PERIOD"), None)
    value_col = next((c for c in raw.columns if c.upper() == "OBS_VALUE"), None)
    if date_col is None or value_col is None:
        raise FetchError(
            f"unexpected ECB {name} CSV columns: {raw.columns}; "
            f"expected TIME_PERIOD + OBS_VALUE"
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


def fetch_estr(start: date) -> pl.DataFrame:
    """Fetch €STR daily fixings from ECB Data Portal, return [date, rate_pct]."""
    return _fetch_ecb_csv(ECB_ESTR_URL, start, name="€STR")


def fetch_eonia(start: date) -> pl.DataFrame:
    """Fetch EONIA daily fixings from ECB Data Portal, return [date, rate_pct].

    EONIA was the eurozone overnight rate from 1999-01-04 until it was
    discontinued on 2022-01-03 (replaced by €STR which started 2019-10-02).
    For backtests starting before €STR's launch, EONIA fills in the pre-2019
    history. Same ACT/360 convention as €STR.
    """
    return _fetch_ecb_csv(ECB_EONIA_URL, start, name="EONIA")


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
