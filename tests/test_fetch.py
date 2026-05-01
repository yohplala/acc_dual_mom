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
    with pytest.raises(fetch.FetchError, match="no data"):
        fetch.fetch_yahoo(_asset(), start=date(2024, 1, 1))


def test_fetch_yahoo_missing_close_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {"Open": [100.0]},
        index=pd.DatetimeIndex(["2024-01-02"]),
    )
    monkeypatch.setattr(yf, "download", lambda *a, **kw: df)
    with pytest.raises(fetch.FetchError, match="missing Close"):
        fetch.fetch_yahoo(_asset(), start=date(2024, 1, 1))


# ── Stooq fetcher tests (HTTP mocked via httpx.Client.get monkeypatch) ──


class _MockResp:
    def __init__(self, content: bytes, status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _MockClient:
    """Drop-in for httpx.Client; stores last params and returns a canned body."""

    def __init__(self, body: bytes, status: int = 200) -> None:
        self._body = body
        self._status = status
        self.last_params: dict[str, str] | None = None

    def __enter__(self) -> _MockClient:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def get(self, url: str, params=None, headers=None) -> _MockResp:
        self.last_params = params
        return _MockResp(self._body, self._status)


def _patch_stooq_response(monkeypatch: pytest.MonkeyPatch, body: bytes) -> _MockClient:
    client = _MockClient(body)
    monkeypatch.setattr(fetch.httpx, "Client", lambda **kw: client)
    return client


_STOOQ_CSV_OK = (
    b"Date,Open,High,Low,Close,Volume\n"
    b"2024-01-02,100.0,101.0,99.5,100.5,12345\n"
    b"2024-01-03,100.5,102.0,100.0,101.8,67890\n"
    b"2024-01-04,101.8,102.5,101.0,102.2,55555\n"
)


def test_fetch_stooq_parses_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_stooq_response(monkeypatch, _STOOQ_CSV_OK)
    out = fetch._fetch_stooq_close_only("^stoxxr", start=date(2024, 1, 1))
    assert out.height == 3
    assert out.schema["date"] == pl.Date
    assert out.get_column("close").to_list() == [100.5, 101.8, 102.2]


def test_fetch_stooq_filters_by_start_date(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_stooq_response(monkeypatch, _STOOQ_CSV_OK)
    out = fetch._fetch_stooq_close_only("^stoxxr", start=date(2024, 1, 3))
    assert out.height == 2
    assert out.get_column("date").to_list() == [date(2024, 1, 3), date(2024, 1, 4)]


def test_fetch_stooq_passes_params_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(fetch.STOOQ_API_KEY_ENV, raising=False)
    client = _patch_stooq_response(monkeypatch, _STOOQ_CSV_OK)
    fetch._fetch_stooq_close_only("^sx5gr", start=date(2020, 5, 15))
    assert client.last_params is not None
    assert client.last_params["s"] == "^sx5gr"
    assert client.last_params["d1"] == "20200515"
    assert client.last_params["i"] == "d"
    assert "apikey" not in client.last_params  # no env → no apikey param


def test_fetch_stooq_appends_apikey_when_env_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(fetch.STOOQ_API_KEY_ENV, "TESTKEY12345")
    client = _patch_stooq_response(monkeypatch, _STOOQ_CSV_OK)
    fetch._fetch_stooq_close_only("^spxtr", start=date(2020, 1, 1))
    assert client.last_params is not None
    assert client.last_params.get("apikey") == "TESTKEY12345"


def test_fetch_stooq_skips_apikey_when_env_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(fetch.STOOQ_API_KEY_ENV, "")
    client = _patch_stooq_response(monkeypatch, _STOOQ_CSV_OK)
    fetch._fetch_stooq_close_only("^spxtr", start=date(2020, 1, 1))
    assert client.last_params is not None
    assert "apikey" not in client.last_params  # empty env treated as unset


def test_fetch_stooq_rejects_non_csv_response(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_stooq_response(monkeypatch, b"<html><body>Captcha required</body></html>")
    with pytest.raises(fetch.FetchError, match="non-CSV"):
        fetch._fetch_stooq_close_only("^bogus", start=date(2024, 1, 1))


def test_fetch_stooq_drops_null_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    body = (
        b"Date,Open,High,Low,Close,Volume\n"
        b"2024-01-02,100.0,101.0,99.5,100.5,12345\n"
        b"2024-01-03,,,,,0\n"  # null close
        b"2024-01-04,101.8,102.5,101.0,102.2,55555\n"
    )
    _patch_stooq_response(monkeypatch, body)
    out = fetch._fetch_stooq_close_only("^stoxxr", start=date(2024, 1, 1))
    assert out.height == 2
    assert out.get_column("close").to_list() == [100.5, 102.2]


# ── Proxy fallback chain (Stooq primary → Yahoo fallback) ──


def _eur_asset_with_fallback() -> Asset:
    return Asset(
        id="eu_test",
        isin="x",
        yahoo="X.PA",
        region="europe",
        inception=date(2018, 1, 1),
        index_proxy="^stoxxr",
        index_proxy_kind="eur_pr",
        index_proxy_source="stooq",
        index_proxy_fallback="^STOXX",
    )


def test_proxy_fallback_engages_when_stooq_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """If Stooq returns an error and a fallback is configured, fall back to Yahoo."""
    asset = _eur_asset_with_fallback()
    _patch_stooq_response(monkeypatch, b"<html>captcha</html>")  # Stooq fails

    yahoo_called: list[str] = []

    def fake_yahoo_close_only(ticker: str, start: date) -> pl.DataFrame:
        yahoo_called.append(ticker)
        return pl.DataFrame({"date": [date(2024, 1, 2), date(2024, 1, 3)], "close": [100.0, 101.0]})

    monkeypatch.setattr(fetch, "_fetch_yahoo_close_only", fake_yahoo_close_only)
    out = fetch._fetch_proxy_in_eur(asset, start=date(2024, 1, 1))
    assert out.height == 2
    assert "^STOXX" in yahoo_called  # fallback was queried


def test_proxy_no_fallback_propagates_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a fallback, Stooq failures bubble up loudly."""
    asset = Asset(
        id="x",
        isin="x",
        yahoo="X.PA",
        region="europe",
        inception=date(2018, 1, 1),
        index_proxy="^stoxxr",
        index_proxy_kind="eur_pr",
        index_proxy_source="stooq",
        index_proxy_fallback=None,  # no fallback
    )
    _patch_stooq_response(monkeypatch, b"<html>captcha</html>")
    with pytest.raises(fetch.FetchError, match="non-CSV"):
        fetch._fetch_proxy_in_eur(asset, start=date(2024, 1, 1))
