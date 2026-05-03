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

Bad-data scrubbing: yfinance has been observed to return half-priced or
doubled close values on isolated dates for `^XXX` indices and EURUSD=X.
The bug pattern is a single-day "round-trip spike" (price drops 50%, then
returns to trend the next day, or symmetric). We detect that pattern and
null-out the corrupt day so the daily-return chain shows 0% on the bad day
and the real return on the recovery day — preserving total return and
correcting vol/Sharpe (instead of inflating them with spurious daily
moves). Genuine non-round-trip outliers above the plausibility ceiling
raise `FetchError` for human investigation.
"""

from __future__ import annotations

import logging
from datetime import date

import polars as pl

from .errors import FetchError

log = logging.getLogger(__name__)

# Hard ceiling on plausible single-day moves for any equity index or ETF.
# Worst real single-day move on a major equity index since 1928 is Black
# Monday 1987 at -22.6%. COVID 2020 worst day was -12%. Anything beyond
# this threshold is bad data — yfinance has been observed to return
# half-priced or doubled values on isolated dates for both FX series
# (EURUSD=X) and `^XXX` index tickers (^GSPC, ^STOXX50E).
MAX_PLAUSIBLE_DAILY_RETURN = 0.30

# Round-trip-spike detection threshold (separate from the plausibility
# ceiling above). A round-trip is a stretch where close jumps by more
# than this threshold and then snaps back to within `threshold/2` of
# the pre-jump anchor within `max_spike_days`. The world-proxy bad-day
# pattern recurs around European holidays at magnitudes 6-15% with
# clean V-shape recoveries; real high-vol days (Black Monday -22.6%,
# COVID -8.9%, Lehman -8.8%) do NOT V-shape — recoveries are gradual
# over weeks, never returning to the pre-crash level within 3 days.
# Setting the trigger at 6% with the tight 3% recovery requirement
# catches the corruption pattern across the full 6-15% range while
# leaving real moves untouched.
ROUND_TRIP_THRESHOLD = 0.06

# Minimum length of an exact-equal close-value run to be considered a
# forward-fill artefact. Real EUR-denominated equity / index closes
# always change between trading days (even if by 0.01); a stretch of
# >= 3 consecutive identical decimals is statistically implausible.
# Some legacy data on the prices-data branch has 162-day flat runs
# from a previous version of the FX-conversion path; this catches them.
MIN_SUSPICIOUS_FLAT_RUN = 3


def scrub_long_format(
    prices_long: pl.DataFrame,
    threshold: float = ROUND_TRIP_THRESHOLD,
    min_flat_run: int = MIN_SUSPICIOUS_FLAT_RUN,
    max_spike_days: int = 3,
) -> pl.DataFrame:
    """Backtest-layer defensive scrub for ``[date, asset_id, close, source]``
    long-format prices. Per-asset, sets the ``close`` column to null for any
    day whose value is implausible:

    1. **Round-trip spikes (1 to ``max_spike_days`` days long)** — close
       jumps by more than ``threshold`` (default 15%) from a real-looking
       baseline, stays at the spurious level for up to ``max_spike_days``
       consecutive days, then snaps back to within ``threshold`` of the
       baseline. All days inside the spike interval are nulled; the
       baseline (day before) and the recovery day (day after) are kept.
       The 15% pair-threshold is intentionally below the 30% plausibility
       ceiling because real high-vol days (Black Monday -22.6%, COVID
       2020 -12%) do NOT immediately reverse; only data corruption
       produces the round-trip pattern.

    2. **Sustained flat runs** — ``min_flat_run`` or more consecutive
       trading days with the *exact* same close (to full float
       precision). This is the signature of forward-fill artefacts in
       the historical fetch pipeline — real equity prices never sit at
       identical decimal values across multi-day stretches.

    Backtest's existing ``forward_fill`` on the wide-format pivot
    absorbs the nulls (carrying the previous valid close), so the
    asset's daily return becomes 0% on the bad days and resumes
    normally on the next valid day. Total return preserved, vol /
    Sharpe / Max DD corrected.

    The safe asset is exempt from the flat-run check (synthetic series
    can theoretically have very small or zero rate days, though in
    practice €STR/EONIA always vary).
    """
    if prices_long.is_empty():
        return prices_long

    sorted_df = prices_long.sort(["asset_id", "date"])

    # 1. Round-trip detection: per-asset Python loop. The polars-native
    #    shift expressions can detect single-day spikes only; multi-day
    #    spikes (e.g. world had a 2-day pattern Mar 29 + Apr 1, recovery
    #    Apr 2) need an iterative scan.
    is_round_trip: list[bool] = [False] * sorted_df.height
    pos = 0
    for _asset_id, group_df in sorted_df.group_by("asset_id", maintain_order=True):
        closes = group_df.get_column("close").to_list()
        for k in _detect_round_trip_indices(closes, threshold, max_spike_days):
            is_round_trip[pos + k] = True
        pos += len(closes)

    # 2. Flat-run detection: vectorised in polars. Skip for the safe asset.
    cleaned = (
        sorted_df.with_columns(
            _is_round_trip=pl.Series(is_round_trip, dtype=pl.Boolean),
            _same_as_prev=(pl.col("close") == pl.col("close").shift(1).over("asset_id")).fill_null(
                False
            ),
        )
        .with_columns(
            _run_id=(~pl.col("_same_as_prev")).cum_sum().over("asset_id"),
        )
        .with_columns(
            _run_size=pl.len().over(["asset_id", "_run_id"]),
        )
        .with_columns(
            _is_flat_run=((pl.col("_run_size") >= min_flat_run) & (pl.col("asset_id") != "safe")),
        )
        .with_columns(
            _is_bad=(pl.col("_is_round_trip") | pl.col("_is_flat_run")),
        )
    )

    n_round_trip = cleaned.filter(pl.col("_is_round_trip")).height
    n_flat_run = cleaned.filter(pl.col("_is_flat_run")).height
    if n_round_trip + n_flat_run > 0:
        per_asset = (
            cleaned.filter(pl.col("_is_bad"))
            .group_by("asset_id")
            .agg(
                pl.col("_is_round_trip").sum().alias("rt"),
                pl.col("_is_flat_run").sum().alias("ff"),
            )
            .sort("asset_id")
        )
        details = "; ".join(
            f"{r['asset_id']} (round-trip={r['rt']}, flat={r['ff']})"
            for r in per_asset.iter_rows(named=True)
        )
        log.warning(
            "scrub_long_format: nulled %d round-trip spike(s) + %d flat-run day(s). %s",
            n_round_trip,
            n_flat_run,
            details,
        )

    return cleaned.with_columns(
        close=pl.when(pl.col("_is_bad")).then(None).otherwise(pl.col("close")),
    ).drop(
        "_is_round_trip",
        "_same_as_prev",
        "_run_id",
        "_run_size",
        "_is_flat_run",
        "_is_bad",
    )


def _detect_round_trip_indices(
    closes: list[float | None], threshold: float, max_spike_days: int
) -> set[int]:
    """Return the set of indices inside round-trip spikes of length 1 to
    ``max_spike_days``. A "spike" is a stretch where close jumps by more
    than ``threshold`` away from the day-before baseline (anchor), stays
    at the spurious level, then snaps back to within ``threshold/2`` of
    the anchor within ``max_spike_days``. The asymmetric trigger/recovery
    thresholds discriminate corrupt-data spikes (full V-shape recovery
    within 1-3 days) from real high-vol days (recovery is always gradual
    — Black Monday 1987 -22.6% recovered only 5% the next day, COVID
    2020 -8.9% recovered 1.6%, Lehman -8.8% recovered 5.4%). The
    baseline day and the recovery day are NOT flagged — only the
    corrupt days in between.
    """
    bad: set[int] = set()
    recovery_threshold = threshold / 2.0
    n = len(closes)
    i = 1
    while i < n:
        c_prev = closes[i - 1]
        c_t = closes[i]
        if c_t is None or c_prev is None or c_prev <= 0:
            i += 1
            continue
        if abs(c_t / c_prev - 1.0) <= threshold:
            i += 1
            continue
        # Big jump on day i — look for V-shape recovery within max_spike_days.
        recovery_idx: int | None = None
        for k in range(1, max_spike_days + 1):
            j = i + k
            if j >= n:
                break
            c_j = closes[j]
            if c_j is None or c_j <= 0:
                continue
            if abs(c_j / c_prev - 1.0) < recovery_threshold:
                recovery_idx = j
                break
        if recovery_idx is not None:
            for k in range(i, recovery_idx):
                bad.add(k)
            i = recovery_idx
        else:
            # No tight V-shape recovery within max_spike_days — sustained
            # move (real high-vol period, or long bad-data period). Don't
            # scrub here; flat-run detection handles the long bad-data
            # case, and real moves pass through unchanged.
            i += 1
    return bad


def _scrub_round_trip_spikes(
    series: pl.DataFrame,
    asset_id: str,
    label: str,
    threshold: float = MAX_PLAUSIBLE_DAILY_RETURN,
) -> pl.DataFrame:
    """Null-out single-day round-trip spikes in a `[date, close]` series.

    A "round-trip spike" is a day t where the close jumps by more than
    `threshold` (e.g., -50%) AND immediately reverts the next day by a
    similarly large opposite-sign move. yfinance is the only source we've
    seen produce this pattern; the corrupt day is bookended by real data
    on both sides, so nulling close[t] while keeping close[t-1] and
    close[t+1] preserves the real total return and removes the phantom
    daily-vol spike. Backtest pipelines forward-fill close (for cross-
    asset calendar alignment) which makes the nulled day a 0% no-op.

    Logs a warning per scrub so data-quality issues are visible without
    crashing the run. Solo outliers (>threshold but no opposite-sign
    bounce) are NOT scrubbed — they're left for `_validate_returns_or_raise`
    to reject loudly, since they would indicate genuine corruption that
    needs human investigation.
    """
    if series.height < 3:
        return series
    sorted_s = series.sort("date")
    closes = sorted_s.get_column("close").to_list()
    bad_indices: list[int] = []
    for t in range(1, len(closes) - 1):
        c_prev, c_t, c_next = closes[t - 1], closes[t], closes[t + 1]
        if c_prev is None or c_t is None or c_next is None:
            continue
        if c_prev <= 0 or c_t <= 0 or c_next <= 0:
            continue
        ret_in = c_t / c_prev - 1.0
        ret_out = c_next / c_t - 1.0
        if abs(ret_in) > threshold and abs(ret_out) > threshold and ret_in * ret_out < 0:
            bad_indices.append(t)
    if not bad_indices:
        return sorted_s

    sample = [(closes[t], sorted_s.get_column("date")[t]) for t in bad_indices[:5]]
    log.warning(
        "%s %s: scrubbed %d round-trip spike(s) (close set to null, forward-fill "
        "in the backtest absorbs them). First 5: %s",
        asset_id,
        label,
        len(bad_indices),
        "; ".join(f"{d} close={c:.4f}" for c, d in sample),
    )
    new_closes: list[float | None] = list(closes)
    for t in bad_indices:
        new_closes[t] = None
    return sorted_s.with_columns(pl.Series("close", new_closes).alias("close"))


def _validate_returns_or_raise(
    series: pl.DataFrame, asset_id: str, label: str, threshold: float = MAX_PLAUSIBLE_DAILY_RETURN
) -> None:
    """Raise `FetchError` if any single-day return on `series` exceeds the
    plausibility ceiling. Run AFTER `_scrub_round_trip_spikes` so that only
    solo outliers (non-round-trip data corruption that needs human
    investigation) survive to here. `label` identifies the segment for the
    error message ("proxy", "live ETF", or "FX")."""
    if series.height < 2:
        return
    rets = series.sort("date").with_columns(ret=(pl.col("close") / pl.col("close").shift(1) - 1.0))
    bad = rets.filter(pl.col("ret").abs() > threshold)
    if bad.is_empty():
        return
    sample = bad.select(["date", "close", "ret"]).head(5).rows()
    raise FetchError(
        f"splice {asset_id}: {label} series has {bad.height} non-round-trip day(s) "
        f"with |return| > {threshold:.0%} — implausible for any equity index, "
        f"strongly suggests bad upstream data that didn't bounce back the next day "
        f"(typical yfinance round-trip spikes are scrubbed automatically; this is "
        f"the residual that needs human investigation). First 5: "
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

    # Two-stage data-quality pass:
    # 1. Scrub round-trip single-day spikes (the typical yfinance bad-day
    #    pattern: -50% then +100%, or symmetric). The bad close is set to
    #    null; downstream forward-fill in backtest.run() absorbs it,
    #    preserving the real total return while removing the phantom vol.
    # 2. After scrubbing, check for any remaining solo outliers (>30%
    #    moves that DIDN'T bounce back). Those indicate genuine
    #    corruption requiring investigation; raise FetchError.
    proxy_long = _scrub_round_trip_spikes(proxy_long, asset_id, "proxy")
    etf_long = _scrub_round_trip_spikes(etf_long, asset_id, "live ETF")
    _validate_returns_or_raise(proxy_long, asset_id, "proxy")
    _validate_returns_or_raise(etf_long, asset_id, "live ETF")

    # Splice date = ETF's first real close. The `>= inception` filter is
    # a defensive floor only: yfinance occasionally returns synthetic /
    # zero-volume rows BEFORE an ETF's official launch (renamed share
    # classes, backfilled NAV records — C50.PA hit us with this). The
    # configured `inception` says "treat anything Yahoo gives us before
    # this as bogus"; the splice anchor is the first surviving row.
    #
    # Many Amundi PEA ETFs also have a multi-week or multi-month *delay*
    # between official launch and the first day Yahoo carries data for
    # them (PSP5.PA: 22 days, PAASI.PA: 128 days, PCEU.PA: 135 days).
    # Anchoring the splice at the configured inception in those cases
    # would truncate the proxy at inception-1 and leave a hole until live
    # data starts; driving the cut-over off the actual first close lets
    # the proxy carry the series through the dark window with no gap.
    etf_at = etf_long.filter(pl.col("date") >= inception).sort("date").head(1)
    if etf_at.is_empty():
        raise FetchError(f"splice {asset_id}: no ETF data on or after inception date {inception}")
    splice_date = etf_at.get_column("date")[0]

    proxy_at = proxy_long.filter(pl.col("date") <= splice_date).sort("date").tail(1)
    if proxy_at.is_empty():
        raise FetchError(
            f"splice {asset_id}: no proxy data on or before splice date {splice_date} "
            f"(proxy series begins after the splice point)"
        )

    etf_close = float(etf_at.get_column("close")[0])
    proxy_close = float(proxy_at.get_column("close")[0])
    if proxy_close <= 0:
        raise FetchError(
            f"splice {asset_id}: proxy close at splice date is non-positive ({proxy_close})"
        )
    scale = etf_close / proxy_close

    pre = (
        proxy_long.filter(pl.col("date") < splice_date)
        .sort("date")
        .with_columns(
            close=pl.col("close") * scale,
            asset_id=pl.lit(asset_id),
            source=pl.lit("stitched_index_proxy"),
        )
        .select(["date", "asset_id", "close", "source"])
    )

    # ETF segment starts at the splice date; any phantom pre-inception
    # rows have already been excluded by the >= inception filter when
    # we picked etf_at, so they can't contaminate the post side either.
    post = etf_long.filter(pl.col("date") >= splice_date).select(
        ["date", "asset_id", "close", "source"]
    )
    return pl.concat([pre, post]).sort("date")


def usd_to_eur(idx_usd: pl.DataFrame, eurusd: pl.DataFrame) -> pl.DataFrame:
    """Convert a USD-denominated `[date, close]` series to EUR using a
    daily `[date, close]` EURUSD series (USD per EUR). EUR price = USD / EURUSD.
    Raises `FetchError` if either input is empty.
    """
    return _convert(idx_usd, eurusd, ccy="USD")


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

    # Scrub round-trip FX spikes first, then loud-fail on residual solo
    # outliers. A bad EURUSD=X day silently corrupts every cross-currency
    # proxy by exactly the same factor, so catching it here pinpoints the
    # cause (worst-ever real EURUSD daily move ≈ 4-5%, well below 30%).
    fx_per_eur = _scrub_round_trip_spikes(fx_per_eur, f"EUR{ccy}=X", "FX")
    _validate_returns_or_raise(fx_per_eur, asset_id=f"EUR{ccy}=X", label="FX")

    fx = fx_per_eur.select(["date", pl.col("close").alias("_fx")])
    # Left join + forward-fill on the FX side: the index calendar (US for
    # USD-source proxies) and the FX calendar (TARGET-2 for ECB EUR/USD)
    # diverge a few times a year — Easter Monday, May 1, Dec 24 are
    # TARGET-2 holidays but US-open. An inner join would silently drop
    # those rows from the ETF series, accumulating ~5 phantom missing
    # days per year over a 17-year backtest. Forward-filling the most
    # recent TARGET-2 fixing into US-only days matches market practice
    # (a bank rolling a holiday position uses the prior fixing) and
    # injects no spurious volatility (FX is held flat across the
    # mismatch day, so the EUR return is the local USD return on that
    # day, untouched by FX).
    joined = idx_local.join(fx, on="date", how="left").sort("date")
    joined = joined.with_columns(pl.col("_fx").forward_fill())
    joined = joined.filter(
        (pl.col("_fx") > 0) & pl.col("_fx").is_not_nan() & pl.col("close").is_not_nan()
    )
    if joined.is_empty():
        raise FetchError(
            f"{ccy}→EUR: no overlapping dates between index and FX series after positivity filter"
        )
    return joined.with_columns(close=pl.col("close") / pl.col("_fx")).select(["date", "close"])
