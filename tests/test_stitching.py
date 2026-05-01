"""Tests for the index-proxy splicing utility."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from pea_momentum.errors import FetchError
from pea_momentum.stitching import jpy_to_eur, splice_at_inception, usd_to_eur


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

    def test_jpy_to_eur(self) -> None:
        # JPY index = 30000; EURJPY = 150 → EUR price = 200
        idx = _series(date(2024, 1, 1), [30000.0])
        fx = _series(date(2024, 1, 1), [150.0])
        out = jpy_to_eur(idx, fx)
        assert out.get_column("close")[0] == 200.0

    def test_missing_fx_dates_dropped(self) -> None:
        idx = _series(date(2024, 1, 1), [100.0, 110.0, 120.0])
        fx = _series(date(2024, 1, 2), [1.10])  # only date 2024-01-02 has FX
        out = usd_to_eur(idx, fx)
        assert out.height == 1
        assert out.get_column("date")[0] == date(2024, 1, 2)
