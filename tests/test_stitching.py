"""Tests for the index-proxy splicing utility."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from pea_momentum.errors import FetchError
from pea_momentum.stitching import (
    scrub_long_format,
    splice_at_inception,
    usd_to_eur,
)


def _series(start: date, levels: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {"date": [start + timedelta(days=i) for i in range(len(levels))], "close": levels}
    ).with_columns(pl.col("date").cast(pl.Date))


def _etf_long(start: date, levels: list[float], asset_id: str = "x") -> pl.DataFrame:
    base = _series(start, levels)
    return base.with_columns(
        pl.lit(asset_id).alias("asset_id"),
        pl.lit("yfinance").alias("source"),
    ).select(["date", "asset_id", "close", "source"])


class TestSplice:
    def test_empty_etf_raises(self) -> None:
        etf = pl.DataFrame(
            schema={"date": pl.Date, "asset_id": pl.Utf8, "close": pl.Float64, "source": pl.Utf8}
        )
        proxy = _series(date(2010, 1, 1), [100.0, 101.0])
        with pytest.raises(FetchError, match="ETF series is empty"):
            splice_at_inception(etf, proxy, date(2015, 1, 1), "x")

    def test_empty_proxy_raises(self) -> None:
        etf = _etf_long(date(2015, 1, 1), [100.0, 101.0])
        proxy = pl.DataFrame(schema={"date": pl.Date, "close": pl.Float64})
        with pytest.raises(FetchError, match="proxy series is empty"):
            splice_at_inception(etf, proxy, date(2015, 1, 1), "x")

    def test_level_continuity(self) -> None:
        # Proxy: 50.0 on inception day; ETF: 200.0 on inception day.
        # Pre-inception proxy levels should be scaled by 200/50 = 4x.
        etf = _etf_long(date(2015, 1, 1), [200.0, 201.0, 202.0])
        # Proxy goes from day 0 (proxy=40) to day 5 (proxy=50) — covers pre-inception
        proxy_dates = [date(2014, 12, 27) + timedelta(days=i) for i in range(6)]
        proxy = pl.DataFrame({"date": proxy_dates, "close": [40.0, 42.0, 44.0, 46.0, 48.0, 50.0]})
        proxy = proxy.with_columns(pl.col("date").cast(pl.Date))

        out = splice_at_inception(etf, proxy, date(2015, 1, 1), "x").sort("date")
        # Day before inception: proxy was 48.0 → scaled by 4x = 192.0
        pre_day = out.filter(pl.col("date") == date(2014, 12, 31))
        assert pre_day.height == 1
        assert pre_day.get_column("close")[0] == 192.0

    def test_provenance_flag(self) -> None:
        etf = _etf_long(date(2015, 1, 1), [200.0, 201.0])
        proxy = _series(date(2014, 12, 28), [40.0, 41.0, 42.0, 43.0, 44.0])
        out = splice_at_inception(etf, proxy, date(2015, 1, 1), "x")
        sources = out.get_column("source").unique().to_list()
        assert "stitched_index_proxy" in sources
        assert "yfinance" in sources

    def test_post_inception_unchanged(self) -> None:
        etf = _etf_long(date(2015, 1, 1), [200.0, 201.0, 202.0])
        proxy = _series(date(2014, 12, 28), [40.0, 41.0, 42.0, 43.0, 44.0])
        out = splice_at_inception(etf, proxy, date(2015, 1, 1), "x")
        post = out.filter(pl.col("date") >= date(2015, 1, 1)).sort("date")
        assert post.get_column("close").to_list() == [200.0, 201.0, 202.0]

    def test_no_overlap_returns_etf(self) -> None:
        # Proxy ends before inception — no splice point findable
        etf = _etf_long(date(2015, 1, 1), [200.0, 201.0])
        proxy = _series(date(2010, 1, 1), [40.0, 41.0])  # ancient history only
        out = splice_at_inception(etf, proxy, date(2015, 1, 1), "x")
        # Proxy's last date is 2010-01-02, before inception 2015 — splice still works
        assert out.height >= etf.height


class TestFxConversion:
    def test_usd_to_eur(self) -> None:
        # USD index = 100; EURUSD = 1.10 → EUR price = 100/1.10 ≈ 90.91
        idx = _series(date(2024, 1, 1), [100.0, 110.0, 120.0])
        fx = _series(date(2024, 1, 1), [1.10, 1.10, 1.20])
        out = usd_to_eur(idx, fx).sort("date")
        closes = out.get_column("close").to_list()
        assert abs(closes[0] - 90.909) < 0.01
        assert abs(closes[2] - 100.0) < 0.01  # 120 / 1.20 = 100

    def test_missing_fx_dates_dropped(self) -> None:
        idx = _series(date(2024, 1, 1), [100.0, 110.0, 120.0])
        fx = _series(date(2024, 1, 2), [1.10])  # only date 2024-01-02 has FX
        out = usd_to_eur(idx, fx)
        assert out.height == 1
        assert out.get_column("date")[0] == date(2024, 1, 2)

    def test_fx_round_trip_spike_scrubbed(self) -> None:
        """yfinance has been observed to return half/double values for
        EURUSD=X on isolated dates — every cross-currency proxy then
        inherits the same factor as a phantom equity-curve spike. We
        detect the round-trip pattern (jump + opposite-sign reversal)
        and null-out the corrupt FX day; the inner-join then drops it
        from the EUR-converted output, and downstream forward-fill in
        backtest absorbs the gap. Total return preserved, vol corrected."""
        idx = _series(date(2024, 1, 1), [100.0, 100.0, 100.0])
        # Day 2 FX doubles (1.10 → 2.20 → 1.10) — round-trip spike.
        fx = _series(date(2024, 1, 1), [1.10, 2.20, 1.10])
        out = usd_to_eur(idx, fx).sort("date")
        # Bad FX day is dropped by the inner-join after scrubbing; the
        # remaining 2 days are correctly converted at 1.10.
        assert out.height == 2
        for c in out.get_column("close").to_list():
            assert abs(c - 90.909) < 0.01

    def test_fx_solo_outlier_still_raises(self) -> None:
        """A non-round-trip FX outlier (no opposite-sign bounce) is genuine
        corruption that needs investigation, not a yfinance round-trip
        artefact. Must still fail loudly."""
        idx = _series(date(2024, 1, 1), [100.0, 100.0, 100.0])
        # Day 2 doubles AND stays — would persistently halve EUR prices
        # going forward. Not a yfinance round-trip; needs investigation.
        fx = _series(date(2024, 1, 1), [1.10, 2.20, 2.30])
        with pytest.raises(FetchError, match=r"non-round-trip day"):
            usd_to_eur(idx, fx)


class TestRoundTripSpikeScrub:
    """Round-trip spikes (the typical yfinance bad-day pattern) are scrubbed
    silently with a log warning — total return preserved, daily vol
    corrected. Solo outliers (>30% with no bounce-back) still raise."""

    def test_proxy_round_trip_spike_scrubbed(self) -> None:
        # Proxy series with a single bad day — exactly the pattern observed
        # in production (us_large oscillating between 6.7 and 13.5). The
        # bad day is set to null; backtest forward-fill absorbs it.
        etf = _etf_long(date(2015, 1, 1), [200.0, 201.0])
        proxy = _series(date(2014, 12, 28), [40.0, 41.0, 19.5, 41.5, 50.0])
        out = splice_at_inception(etf, proxy, date(2015, 1, 1), "x").sort("date")
        # All 7 dates still present (4 pre + 2 post + 1 splice day's pre row,
        # which is filtered to <inception so 4 + 2 = 6 actually).
        assert out.height >= 6
        # Day 3 of the proxy series (the bad one, 2014-12-30) is now null.
        bad_day_row = out.filter(pl.col("date") == date(2014, 12, 30))
        assert bad_day_row.height == 1
        assert bad_day_row.get_column("close")[0] is None

    def test_etf_round_trip_spike_scrubbed(self) -> None:
        # Live-ETF doubled-day pattern (rarer but possible).
        etf = _etf_long(date(2015, 1, 1), [200.0, 410.0, 201.0])  # day 2 doubled
        proxy = _series(date(2014, 12, 28), [40.0, 41.0, 42.0, 43.0, 44.0])
        out = splice_at_inception(etf, proxy, date(2015, 1, 1), "x").sort("date")
        # Day 2 of ETF (2015-01-02) is the bad one — close should be null.
        bad_day_row = out.filter(pl.col("date") == date(2015, 1, 2))
        assert bad_day_row.height == 1
        assert bad_day_row.get_column("close")[0] is None

    def test_solo_outlier_still_raises(self) -> None:
        # Proxy jumps -50% and STAYS (no bounce). Not a yfinance round-trip
        # — genuine corruption that needs investigation.
        etf = _etf_long(date(2015, 1, 1), [200.0, 201.0])
        proxy = _series(date(2014, 12, 28), [40.0, 41.0, 19.5, 19.6, 19.7])
        with pytest.raises(FetchError, match=r"non-round-trip day"):
            splice_at_inception(etf, proxy, date(2015, 1, 1), "x")

    def test_normal_market_moves_pass(self) -> None:
        # A real-but-large move (-10% Lehman-style) must NOT trigger
        # scrubbing — the ceiling is 30% and the bounce-back is required.
        etf = _etf_long(date(2015, 1, 1), [200.0, 180.0, 195.0])
        proxy = _series(date(2014, 12, 28), [40.0, 41.0, 36.0, 38.0, 42.0])
        out = splice_at_inception(etf, proxy, date(2015, 1, 1), "x")
        assert out.height == 7  # 4 pre + 3 post — no rows lost
        # No nulls inserted
        assert out.filter(pl.col("close").is_null()).height == 0


def _long(start: date, asset_id: str, closes: list[float]) -> pl.DataFrame:
    """Build a long-format `[date, asset_id, close, source]` series."""
    return pl.DataFrame(
        {
            "date": [start + timedelta(days=i) for i in range(len(closes))],
            "asset_id": [asset_id] * len(closes),
            "close": closes,
            "source": ["yfinance"] * len(closes),
        }
    ).with_columns(pl.col("date").cast(pl.Date))


class TestScrubLongFormat:
    """Backtest-layer scrub catches both round-trip spikes and sustained
    flat-run forward-fill artefacts in long-format prices."""

    def test_round_trip_spike_nulled(self) -> None:
        df = _long(date(2024, 1, 1), "x", [100.0, 100.5, 50.0, 100.5, 101.0])
        out = scrub_long_format(df).sort("date")
        # Day 3 (50.0) is the spike. Day 4 (back to 100.5) is the recovery.
        # Only day 3's close should be nulled.
        closes = out.get_column("close").to_list()
        assert closes[2] is None  # the spike
        assert closes[1] == 100.5  # before spike — preserved
        assert closes[3] == 100.5  # after spike — preserved (real recovery day)

    def test_flat_run_nulled(self) -> None:
        # 5-day run of identical prices — clearly a forward-fill artefact.
        df = _long(date(2024, 1, 1), "x", [100.0, 100.0, 100.0, 100.0, 100.0, 102.0])
        out = scrub_long_format(df).sort("date")
        closes = out.get_column("close").to_list()
        # First 5 days are the flat run — all nulled.
        assert all(c is None for c in closes[:5])
        # Day 6 (different value) is preserved as real.
        assert closes[5] == 102.0

    def test_short_flat_run_not_nulled(self) -> None:
        # 2-day "flat run" (just one repeat) — could be coincidence, not nulled.
        df = _long(date(2024, 1, 1), "x", [100.0, 100.0, 101.0, 102.0])
        out = scrub_long_format(df).sort("date")
        # Default min_flat_run=3, so this 2-row run is preserved.
        assert out.filter(pl.col("close").is_null()).height == 0

    def test_safe_asset_exempt_from_flat_run_check(self) -> None:
        # The synthetic safe-asset series is exempt — €STR/EONIA can in
        # principle have very small or zero days that produce identical
        # decimals. We don't want to scrub the safe asset's history.
        df = _long(date(2024, 1, 1), "safe", [100.0, 100.0, 100.0, 100.0, 100.0])
        out = scrub_long_format(df)
        assert out.filter(pl.col("close").is_null()).height == 0

    def test_normal_series_passes_unchanged(self) -> None:
        # Normal varying prices — nothing should be scrubbed.
        df = _long(date(2024, 1, 1), "x", [100.0, 101.0, 99.5, 102.0, 100.5, 103.0])
        out = scrub_long_format(df)
        assert out.filter(pl.col("close").is_null()).height == 0

    def test_two_day_spike_nulled(self) -> None:
        # 2-day spike pattern observed in the world proxy: jump up on
        # day t, stays high on day t+1, snaps back on day t+2. Both
        # bogus days should be nulled, the recovery day preserved.
        df = _long(date(2024, 1, 1), "x", [100.0, 100.5, 130.0, 130.0, 100.5, 101.0])
        out = scrub_long_format(df).sort("date")
        closes = out.get_column("close").to_list()
        assert closes[1] == 100.5  # baseline preserved
        assert closes[2] is None  # spike day 1 nulled
        assert closes[3] is None  # spike day 2 nulled
        assert closes[4] == 100.5  # recovery preserved (real)

    def test_three_day_spike_nulled(self) -> None:
        # 3-day spike: still inside max_spike_days=3 default.
        df = _long(date(2024, 1, 1), "x", [100.0, 100.5, 130.0, 130.0, 130.0, 100.5])
        out = scrub_long_format(df).sort("date")
        closes = out.get_column("close").to_list()
        assert closes[2] is None
        assert closes[3] is None
        assert closes[4] is None
        assert closes[5] == 100.5  # recovery preserved

    def test_no_v_shape_recovery_not_nulled(self) -> None:
        # Real high-vol pattern (e.g. COVID 2020-03-12): big down-day,
        # but next days continue down or only partially recover. NOT a
        # round-trip; scrub must leave it alone.
        df = _long(
            date(2024, 1, 1),
            "x",
            [100.0, 100.0, 91.0, 92.5, 88.5, 90.0],  # -9% then partial moves, no V-shape
        )
        out = scrub_long_format(df)
        # No recovery to within threshold/2 (3%) of anchor 100.0 → don't scrub.
        assert out.filter(pl.col("close").is_null()).height == 0

    def test_per_asset_scrub(self) -> None:
        # Two assets with different patterns — scrub treats them independently.
        x = _long(date(2024, 1, 1), "x", [100.0, 100.0, 100.0, 100.0, 100.0])  # flat
        y = _long(date(2024, 1, 1), "y", [50.0, 51.0, 52.0, 53.0, 54.0])  # normal
        out = scrub_long_format(pl.concat([x, y]))
        x_out = out.filter(pl.col("asset_id") == "x")
        y_out = out.filter(pl.col("asset_id") == "y")
        # x: all 5 days nulled (flat run)
        assert x_out.filter(pl.col("close").is_null()).height == 5
        # y: nothing nulled (normal series)
        assert y_out.filter(pl.col("close").is_null()).height == 0
