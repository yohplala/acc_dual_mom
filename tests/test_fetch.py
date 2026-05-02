"""Tests for the fetch boundary layer.

We don't hit the network — yfinance and httpx are monkeypatched. The point
of these tests is to lock down the type-conversion path between pandas
(which under pandas 3 may use the nullable Float64 extension dtype) and
polars (which requires pyarrow for that conversion unless we go through
native numpy arrays first).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import polars as pl
import pytest
import yfinance as yf

from pea_momentum import fetch
from pea_momentum.errors import FetchError
from pea_momentum.universe import Asset


def _asset() -> Asset:
    return Asset(id="t", isin="x", yahoo="X.PA", region="x")


def test_fetch_yahoo_handles_pandas_extension_float64(monkeypatch: pytest.MonkeyPatch) -> None:
    """pandas 3 returns Float64 (nullable) by default — must not require pyarrow."""
    df = pd.DataFrame(
        {"Close": pd.array([100.0, 101.5, 102.0], dtype="Float64")},
        index=pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    monkeypatch.setattr(yf, "download", lambda *a, **kw: df)

    result = fetch.fetch_yahoo(_asset(), start=date(2024, 1, 1))

    assert result.height == 3
    assert result.schema["date"] == pl.Date
    assert result.schema["close"] == pl.Float64
    assert result.get_column("close").to_list() == [100.0, 101.5, 102.0]
    assert result.get_column("asset_id").unique().to_list() == ["t"]


@pytest.mark.parametrize("resolution", ["s", "ms", "us", "ns"])
def test_fetch_yahoo_handles_all_datetime_resolutions(
    monkeypatch: pytest.MonkeyPatch, resolution: str
) -> None:
    """Regression: yfinance under pandas 3 yields `datetime64[s]` indices, which
    polars rejects (only D/ms/us/ns are accepted). The fetcher must normalize
    every resolution it might receive."""
    raw_dates = np.array(
        ["2024-01-02", "2024-01-03", "2024-01-04"], dtype=f"datetime64[{resolution}]"
    )
    df = pd.DataFrame(
        {"Close": [100.0, 101.0, 102.0]},
        index=pd.DatetimeIndex(raw_dates),
    )
    monkeypatch.setattr(yf, "download", lambda *a, **kw: df)

    result = fetch.fetch_yahoo(_asset(), start=date(2024, 1, 1))

    assert result.height == 3
    assert result.schema["date"] == pl.Date
    assert result.get_column("date").to_list() == [
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
    ]


def test_fetch_yahoo_drops_nan_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {"Close": pd.array([100.0, np.nan, 102.0], dtype="Float64")},
        index=pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    monkeypatch.setattr(yf, "download", lambda *a, **kw: df)

    result = fetch.fetch_yahoo(_asset(), start=date(2024, 1, 1))

    assert result.height == 2
    assert result.get_column("close").to_list() == [100.0, 102.0]


def test_fetch_yahoo_handles_multilevel_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    """yfinance ≥ 0.2.40 returns multi-level columns even for a single ticker."""
    df = pd.DataFrame(
        [[100.0], [101.0]],
        index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"]),
        columns=pd.MultiIndex.from_tuples([("Close", "X.PA")]),
    )
    monkeypatch.setattr(yf, "download", lambda *a, **kw: df)

    result = fetch.fetch_yahoo(_asset(), start=date(2024, 1, 1))

    assert result.height == 2
    assert result.get_column("close").to_list() == [100.0, 101.0]


def test_fetch_yahoo_empty_response_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yf, "download", lambda *a, **kw: pd.DataFrame())
    with pytest.raises(FetchError, match="no data"):
        fetch.fetch_yahoo(_asset(), start=date(2024, 1, 1))


def test_fetch_yahoo_missing_close_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {"Open": [100.0]},
        index=pd.DatetimeIndex(["2024-01-02"]),
    )
    monkeypatch.setattr(yf, "download", lambda *a, **kw: df)
    with pytest.raises(FetchError, match="missing Close"):
        fetch.fetch_yahoo(_asset(), start=date(2024, 1, 1))


# ── Proxy-in-EUR dispatch ──


def test_proxy_in_eur_rejects_unsupported_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    asset = Asset(
        id="x",
        isin="x",
        yahoo="X.PA",
        region="europe",
        inception=date(2018, 1, 1),
        index_proxy="^stoxxr",
        index_proxy_kind="eur_pr",  # not in SUPPORTED_PROXY_KINDS
    )
    with pytest.raises(FetchError, match="Unsupported index_proxy_kind"):
        fetch._fetch_proxy_in_eur(asset, start=date(2024, 1, 1))


def test_proxy_in_eur_skips_fx_for_eur_tr(monkeypatch: pytest.MonkeyPatch) -> None:
    asset = Asset(
        id="x",
        isin="x",
        yahoo="X.PA",
        region="europe",
        inception=date(2018, 1, 1),
        index_proxy="IWDA.AS",
        index_proxy_kind="eur_tr",
    )
    fetched: list[str] = []

    def fake_close_only(ticker: str, start: date) -> pl.DataFrame:
        fetched.append(ticker)
        return pl.DataFrame({"date": [date(2024, 1, 2)], "close": [100.0]})

    monkeypatch.setattr(fetch, "_fetch_yahoo_close_only", fake_close_only)
    out = fetch._fetch_proxy_in_eur(asset, start=date(2024, 1, 1))
    assert out.height == 1
    assert fetched == ["IWDA.AS"]  # no FX fetch
