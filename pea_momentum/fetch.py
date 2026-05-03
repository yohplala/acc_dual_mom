"""Price fetching: yfinance for ETFs, ECB Data Portal for €STR.

Boundary module — yfinance returns pandas, we convert to polars on ingress.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Callable
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

# Supported `index_proxy_kind` values.
# `eur_tr` / `usd_tr` encode the FX-conversion path for ETF tickers (whether
# the underlying series is PR or TR depends on the chosen ticker).
# `synth` triggers a recipe-driven construction from other building blocks
# instead of fetching a single Yahoo ticker — see `_SYNTH_PROXY_RECIPES`.
PROXY_KIND_EUR_TR = "eur_tr"
PROXY_KIND_USD_TR = "usd_tr"
PROXY_KIND_SYNTH = "synth"
SUPPORTED_PROXY_KINDS: frozenset[str] = frozenset(
    {PROXY_KIND_EUR_TR, PROXY_KIND_USD_TR, PROXY_KIND_SYNTH}
)


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

    Three configurations are supported:
    - No proxy: just the live ETF.
    - Single proxy via `index_proxy` + `index_proxy_kind`: classic single-stage
      splice.
    - Multi-stage chain via `index_proxy_chain`: each less-clean proxy
      extends the previous (cleaner) one's pre-handoff history. The
      assembled chain is then spliced onto the live ETF at inception.

    Proxy failures propagate as `FetchError` (loud failure) — a configured proxy
    that can't be fetched should be fixed in YAML, not masked by ETF-only
    history.
    """
    etf = fetch_yahoo(asset, start=start)
    if asset.inception is None:
        return etf
    if asset.index_proxy_chain is not None:
        proxy = _fetch_proxy_chain_in_eur(asset, start=start)
    elif asset.index_proxy is not None:
        proxy = _fetch_proxy_in_eur(asset, start=start)
    else:
        return etf
    return stitching.splice_at_inception(etf, proxy, asset.inception, asset.id)


def _fetch_proxy_in_eur(asset: Asset, start: date) -> pl.DataFrame:
    """Fetch the proxy index from Yahoo and convert to EUR if needed.

    Returns `[date, close]`. Raises `FetchError` if the proxy ticker is missing
    or the kind is unsupported.
    """
    if asset.index_proxy is None:
        raise FetchError(f"Asset {asset.id} has no index_proxy configured")
    return _fetch_one_proxy_in_eur(asset.index_proxy, asset.index_proxy_kind, asset.id, start)


def _fetch_one_proxy_in_eur(
    ticker: str, kind: str | None, asset_id: str, start: date
) -> pl.DataFrame:
    """Fetch a single proxy and return `[date, close]` in EUR.

    For `kind=eur_tr` / `usd_tr` the `ticker` is a Yahoo symbol; for
    `kind=synth` the `ticker` names a synthesis recipe in
    `_SYNTH_PROXY_RECIPES` instead of being a real ticker.
    """
    resolved_kind = kind or PROXY_KIND_EUR_TR
    if resolved_kind not in SUPPORTED_PROXY_KINDS:
        raise FetchError(
            f"Unsupported index_proxy_kind {resolved_kind!r} for asset {asset_id} "
            f"(supported: {sorted(SUPPORTED_PROXY_KINDS)})"
        )
    if resolved_kind == PROXY_KIND_SYNTH:
        recipe = _SYNTH_PROXY_RECIPES.get(ticker)
        if recipe is None:
            raise FetchError(
                f"Unknown synthetic-proxy recipe {ticker!r} for asset {asset_id} "
                f"(available: {sorted(_SYNTH_PROXY_RECIPES)})"
            )
        return recipe(start)
    idx = _fetch_yahoo_close_only(ticker, start=start)
    if resolved_kind == PROXY_KIND_EUR_TR:
        return idx
    fx = _fetch_yahoo_close_only("EURUSD=X", start=start)
    return stitching.usd_to_eur(idx, fx)


def _fetch_proxy_chain_in_eur(asset: Asset, start: date) -> pl.DataFrame:
    """Build a multi-stage proxy chain into a single EUR `[date, close]` series.

    `asset.index_proxy_chain` is ordered CLEANEST first (latest start, best
    methodological match) and DIRTIEST last (earliest start, more drift).
    Each less-clean proxy extends the cleaner one's history pre-handoff
    (handoff = cleaner's first available date) by being level-rescaled
    so its close on the handoff date equals the cleaner's close. The
    successively-extended series is returned as one continuous EUR-
    denominated `[date, close]` DataFrame, ready to be passed through
    `splice_at_inception` against the live ETF.

    Each chain element is fetched in EUR (via the same `usd_tr` /
    `eur_tr` paths as a single proxy). Chain elements that have no data
    on or before the handoff date are skipped with a warning — they
    extend nothing and removing them simplifies the cumulative splice.
    """
    chain = asset.index_proxy_chain
    if chain is None or len(chain) == 0:
        raise FetchError(f"Asset {asset.id} has empty index_proxy_chain")

    # Fetch each element in EUR. eur_series[0] = cleanest, eur_series[-1] = dirtiest.
    eur_series: list[pl.DataFrame] = [
        _fetch_one_proxy_in_eur(ticker, kind, asset.id, start) for ticker, kind in chain
    ]

    # Iteratively splice: result starts as cleanest, each successor extends it pre-handoff.
    result = eur_series[0]
    for i in range(1, len(eur_series)):
        nxt = eur_series[i]
        handoff = result.get_column("date").min()
        prev_at = result.filter(pl.col("date") == handoff).head(1)
        # The successor proxy may not have data ON the exact handoff (e.g. holiday in
        # one market, trading day in the other). Take its last close on or before
        # handoff for the rescaling anchor; if none exists, the successor doesn't
        # extend history and we skip it.
        nxt_at = nxt.filter(pl.col("date") <= handoff).sort("date").tail(1)
        if prev_at.is_empty() or nxt_at.is_empty():
            log.warning(
                "proxy chain for %s: skipping segment %d (%s) — no overlap with "
                "predecessor at handoff %s",
                asset.id,
                i,
                chain[i][0],
                handoff,
            )
            continue
        prev_close = float(prev_at.get_column("close")[0])
        nxt_close = float(nxt_at.get_column("close")[0])
        if nxt_close <= 0:
            raise FetchError(
                f"proxy chain {asset.id}: segment {chain[i][0]!r} has non-positive "
                f"close {nxt_close} on or before handoff {handoff!r}"
            )
        scale = prev_close / nxt_close
        pre = nxt.filter(pl.col("date") < handoff).with_columns(close=pl.col("close") * scale)
        if pre.is_empty():
            continue
        result = pl.concat([pre, result]).sort("date")
    return result


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


# ── Synthetic-proxy recipes ───────────────────────────────────────────
# A recipe takes the requested start date and returns a `[date, close]`
# DataFrame in EUR (same shape as a yfinance-fetched proxy). It's an escape
# hatch for proxies that can't be fetched as a single Yahoo ticker — e.g.
# EUR-hedged Japan pre-2010 has no live ETF, so we synthesise it from
# longer-history components.


def _synth_eur_hedged_jp(start: date) -> pl.DataFrame:
    """Synthesise EUR-hedged MSCI Japan from longer-history components.

    Construction:
        L_synth(t) = (EWJ(t) * USDJPY(t)) / (EWJ(0) * USDJPY(0))
                   * (estr(t) / estr(0))

    Decomposition:
        EWJ_USD * USDJPY = (MSCI_Japan_JPY / USDJPY) * USDJPY = MSCI_Japan_JPY
        → JPY-denominated equity level — strips out the EWJ ETF's unhedged
          JPY/USD FX exposure, leaving pure JPY equity returns.
        € carry term `estr / estr(0)` adds the EUR short-rate compounding
        (with EONIA splice pre-2019, handled by `fetch_estr` / `fetch_eonia`),
        which IS the carry a EUR investor earns by selling JPY forward.

    Approximation: the carry term implicitly assumes the JPY short rate is
    zero. Holds tightly pre-2010 (BOJ ZIRP since 1999, so the assumption
    matches reality within a few bps); drifts ~80bp/year against the
    actual IJPE.L (true EUR-hedged Japan) post-2015 because of BOJ's NIRP
    era. The chain mechanism splices the actual IJPE.L on top from
    2010-09-30 onward, so the synthetic only sees use in the pre-2010
    stub where its assumption is best.
    """
    # Components — fetch independently, inner-join on common dates.
    ewj = _fetch_yahoo_close_only("EWJ", start=start).rename({"close": "ewj"})
    usdjpy = _fetch_yahoo_close_only("USDJPY=X", start=start).rename({"close": "usdjpy"})
    # ECB EUR short rate compounded into a price level (EONIA pre-2019 splice
    # already handled by fetch_synth_asset's internal logic). We mimic the
    # same path here without the asset_id/source columns.
    if start >= ESTR_START:
        rates = fetch_estr(start=start)
    else:
        eonia = fetch_eonia(start=start)
        if eonia.is_empty():
            raise FetchError(
                f"synth eur_hedged_jp: EONIA fetch returned empty data — "
                f"cannot construct pre-{ESTR_START} carry term."
            )
        estr = fetch_estr(start=ESTR_START)
        rates = (
            pl.concat([eonia.filter(pl.col("date") < ESTR_START), estr])
            .sort("date")
            .unique(subset=["date"], keep="last")
        )
    if rates.is_empty():
        raise FetchError("synth eur_hedged_jp: ECB EUR short-rate series is empty")
    estr_level = (
        rates.with_columns(daily_factor=(1.0 + pl.col("rate_pct") / 100.0 / 360.0))
        .with_columns(estr=pl.col("daily_factor").cum_prod() * ESTR_BASE_PRICE)
        .select(["date", "estr"])
    )

    joined = ewj.join(usdjpy, on="date", how="inner").join(estr_level, on="date", how="inner")
    if joined.is_empty():
        raise FetchError(
            "synth eur_hedged_jp: no overlapping dates between EWJ, USDJPY, and ECB rate series"
        )
    joined = joined.sort("date")

    # Compose the synthetic level (rebased to the joined-window's first day).
    return joined.with_columns(
        close=(pl.col("ewj") * pl.col("usdjpy"))
        / (pl.col("ewj").first() * pl.col("usdjpy").first())
        * (pl.col("estr") / pl.col("estr").first())
    ).select(["date", "close"])


_SYNTH_PROXY_RECIPES: dict[str, Callable[[date], pl.DataFrame]] = {
    "eur_hedged_jp": _synth_eur_hedged_jp,
}


def _earliest_useful_start() -> date:
    """Default start: 2008-01-01, far enough to include the 2008 GFC for stitched
    strategies. Assets without an `index_proxy` configured will simply have
    no data before their ETF inception — backtests filter to the period when
    all required assets are present."""
    return date(2008, 1, 1)
