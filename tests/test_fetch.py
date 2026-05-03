"""Tests for the fetch boundary layer.

We don't hit the network — yfinance and httpx are monkeypatched. The point
of these tests is to lock down the type-conversion path between pandas
(which under pandas 3 may use the nullable Float64 extension dtype) and
polars (which requires pyarrow for that conversion unless we go through
native numpy arrays first).
"""

from __future__ import annotations

from datetime import date

import httpx
import numpy as np
import pandas as pd
import polars as pl
import pytest
import yfinance as yf

from pea_momentum import fetch
from pea_momentum.errors import FetchError
from pea_momentum.universe import Asset


def _asset() -> Asset:
    return Asset(id="t", isin="x", yahoo="X.PA")


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


# ── Multi-stage proxy chain ──


def test_proxy_chain_extends_pre_handoff_with_rescaled_predecessor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two-stage chain: cleanest covers 2020+, dirtier covers 2018+. After
    chaining, the dirtier proxy's 2018-2019 segment is rescaled so its
    close on the cleanest's first date matches. Continuity at handoff."""
    asset = Asset(
        id="world",
        isin="x",
        yahoo="X.PA",
        inception=date(2025, 1, 1),
        index_proxy_chain=(
            ("CLEANEST", "eur_tr"),  # cleanest, EUR-native, since 2020-01-02
            ("DIRTIER", "eur_tr"),  # extends back to 2018-01-02
        ),
    )

    series = {
        "CLEANEST": pl.DataFrame(
            {
                "date": [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)],
                "close": [200.0, 202.0, 204.0],
            }
        ),
        "DIRTIER": pl.DataFrame(
            {
                "date": [
                    date(2018, 1, 2),
                    date(2019, 1, 2),
                    date(2020, 1, 2),
                    date(2020, 1, 3),
                ],
                "close": [80.0, 90.0, 100.0, 101.0],
            }
        ),
    }
    monkeypatch.setattr(fetch, "_fetch_yahoo_close_only", lambda t, start: series[t])

    out = fetch._fetch_proxy_chain_in_eur(asset, start=date(2018, 1, 1))

    # DIRTIER is rescaled by 200/100 = 2.0 over the pre-handoff segment.
    # Pre-handoff (2018-01-02 + 2019-01-02) → 80*2 = 160, 90*2 = 180.
    # 2020-01-02 onwards comes from CLEANEST: 200, 202, 204.
    assert out.get_column("date").to_list() == [
        date(2018, 1, 2),
        date(2019, 1, 2),
        date(2020, 1, 2),
        date(2020, 1, 3),
        date(2020, 1, 6),
    ]
    assert out.get_column("close").to_list() == [160.0, 180.0, 200.0, 202.0, 204.0]


def test_proxy_chain_skips_segment_with_no_pre_handoff_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chain element whose history doesn't extend before the previous
    element's first date is silently skipped (it can't extend anything)."""
    asset = Asset(
        id="x",
        isin="x",
        yahoo="X.PA",
        inception=date(2025, 1, 1),
        index_proxy_chain=(
            ("CLEAN", "eur_tr"),
            ("USELESS", "eur_tr"),  # starts AFTER CLEAN — should be skipped
        ),
    )
    series = {
        "CLEAN": pl.DataFrame(
            {"date": [date(2020, 1, 2), date(2020, 1, 3)], "close": [100.0, 101.0]}
        ),
        "USELESS": pl.DataFrame(
            {"date": [date(2021, 1, 2), date(2021, 1, 3)], "close": [50.0, 51.0]}
        ),
    }
    monkeypatch.setattr(fetch, "_fetch_yahoo_close_only", lambda t, start: series[t])

    out = fetch._fetch_proxy_chain_in_eur(asset, start=date(2019, 1, 1))
    # Only CLEAN's data is in the output — USELESS contributed nothing.
    assert out.get_column("date").to_list() == [date(2020, 1, 2), date(2020, 1, 3)]
    assert out.get_column("close").to_list() == [100.0, 101.0]


def test_proxy_chain_with_synth_kind_dispatches_to_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chain entry with kind='synth' looks up the recipe in
    `_SYNTH_PROXY_RECIPES` instead of fetching a Yahoo ticker. The recipe
    itself returns a [date, close] DataFrame in EUR, which the splice
    machinery then handles like any other proxy."""
    asset = Asset(
        id="x",
        isin="x",
        yahoo="X.PA",
        inception=date(2025, 1, 1),
        index_proxy_chain=(
            ("CLEAN", "eur_tr"),
            ("toy_recipe", "synth"),  # extends pre-handoff
        ),
    )
    series = {
        "CLEAN": pl.DataFrame(
            {"date": [date(2020, 1, 2), date(2020, 1, 3)], "close": [100.0, 101.0]}
        ),
    }
    recipe_calls: list[date] = []

    def toy_recipe(start: date) -> pl.DataFrame:
        recipe_calls.append(start)
        return pl.DataFrame(
            {
                "date": [date(2018, 1, 2), date(2019, 1, 2), date(2020, 1, 2)],
                "close": [40.0, 45.0, 50.0],
            }
        )

    monkeypatch.setattr(fetch, "_fetch_yahoo_close_only", lambda t, start: series[t])
    monkeypatch.setitem(fetch._SYNTH_PROXY_RECIPES, "toy_recipe", toy_recipe)

    out = fetch._fetch_proxy_chain_in_eur(asset, start=date(2018, 1, 1))

    # Recipe was invoked once with the requested start.
    assert recipe_calls == [date(2018, 1, 1)]
    # Pre-handoff segment is rescaled (50 → 100, scale 2.0): 40→80, 45→90.
    assert out.get_column("date").to_list() == [
        date(2018, 1, 2),
        date(2019, 1, 2),
        date(2020, 1, 2),
        date(2020, 1, 3),
    ]
    assert out.get_column("close").to_list() == [80.0, 90.0, 100.0, 101.0]


def test_synth_proxy_unknown_recipe_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    asset = Asset(
        id="x",
        isin="x",
        yahoo="X.PA",
        inception=date(2025, 1, 1),
        index_proxy="bogus_recipe",
        index_proxy_kind="synth",
    )
    with pytest.raises(FetchError, match="Unknown synthetic-proxy recipe"):
        fetch._fetch_proxy_in_eur(asset, start=date(2020, 1, 1))


def test_synth_eur_hedged_jp_construction_math(monkeypatch: pytest.MonkeyPatch) -> None:
    """The eur_hedged_jp recipe's level formula:
    L(t) = (EWJ(t) * USDJPY(t)) / (EWJ(0) * USDJPY(0)) * (estr(t) / estr(0))

    With deterministic toy inputs we can verify the math exactly."""
    series = {
        "EWJ": pl.DataFrame(
            {
                "date": [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)],
                "close": [50.0, 55.0, 60.0],
            }
        ),
        "USDJPY=X": pl.DataFrame(
            {
                "date": [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)],
                "close": [100.0, 110.0, 120.0],
            }
        ),
    }
    monkeypatch.setattr(fetch, "_fetch_yahoo_close_only", lambda t, start: series[t])
    # Mock €STR rate series — daily 0% for simplicity (level constant at base price).
    monkeypatch.setattr(
        fetch,
        "fetch_estr",
        lambda start: pl.DataFrame(
            {
                "date": [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)],
                "rate_pct": [0.0, 0.0, 0.0],
            }
        ),
    )
    # Skip EONIA splice (start is post-ESTR_START)
    out = fetch._synth_eur_hedged_jp(start=date(2020, 1, 1))
    # Day 1: rebased to 1.0
    # Day 2: (55*110)/(50*100) * 1.0 = 6050/5000 = 1.21
    # Day 3: (60*120)/(50*100) * 1.0 = 7200/5000 = 1.44
    closes = out.get_column("close").to_list()
    assert closes[0] == pytest.approx(1.0)
    assert closes[1] == pytest.approx(1.21)
    assert closes[2] == pytest.approx(1.44)


def test_proxy_chain_handles_handoff_on_non_overlapping_calendar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the dirtier proxy doesn't trade on the exact handoff date (e.g.
    holiday), use its last close on or before the handoff as the rescaling
    anchor."""
    asset = Asset(
        id="x",
        isin="x",
        yahoo="X.PA",
        inception=date(2025, 1, 1),
        index_proxy_chain=(
            ("CLEAN", "eur_tr"),  # starts 2020-01-06 (Monday after Jan 3 Friday)
            ("DIRTIER", "eur_tr"),  # missing 2020-01-06; has 2020-01-03
        ),
    )
    series = {
        "CLEAN": pl.DataFrame({"date": [date(2020, 1, 6)], "close": [120.0]}),
        "DIRTIER": pl.DataFrame(
            {"date": [date(2019, 1, 2), date(2020, 1, 3)], "close": [50.0, 60.0]}
        ),
    }
    monkeypatch.setattr(fetch, "_fetch_yahoo_close_only", lambda t, start: series[t])

    out = fetch._fetch_proxy_chain_in_eur(asset, start=date(2019, 1, 1))
    # Anchor = DIRTIER on 2020-01-03 (last <= handoff 2020-01-06) → 60.
    # Scale = 120 / 60 = 2.0. Pre-handoff: 2019-01-02 → 50*2 = 100, 2020-01-03 → 60*2=120.
    assert out.get_column("date").to_list() == [
        date(2019, 1, 2),
        date(2020, 1, 3),
        date(2020, 1, 6),
    ]
    assert out.get_column("close").to_list() == [100.0, 120.0, 120.0]


# ── ECB EUR/USD fetch + gap validation ───────────────────────────────────


def _fake_ecb_csv_response(rows: list[tuple[str, float | None]]) -> bytes:
    """Build a minimal ECB-shaped CSV with TIME_PERIOD + OBS_VALUE columns.

    `None` OBS_VALUE encodes a TARGET-2 holiday (ECB returns these as
    blank fields). The real ECB response has 30 columns of metadata —
    we include only what `_fetch_ecb_csv` actually parses."""
    lines = ["TIME_PERIOD,OBS_VALUE"]
    for d, v in rows:
        lines.append(f"{d},{'' if v is None else v}")
    return ("\n".join(lines) + "\n").encode()


class _FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None: ...


class _FakeClient:
    def __init__(self, content: bytes) -> None:
        self._content = content

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *_: object) -> None: ...

    def get(self, url: str, headers: dict[str, str] | None = None) -> _FakeResponse:
        return _FakeResponse(self._content)


def test_fetch_eurusd_ecb_returns_eurusd_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """`fetch_eurusd_ecb` returns `[date, close]` with `close = USD per
    1 EUR` — same column shape as the legacy yfinance EURUSD=X output,
    so `stitching.usd_to_eur` consumes it unchanged."""
    csv = _fake_ecb_csv_response(
        [
            ("2024-01-02", 1.10),
            ("2024-01-03", 1.11),
            ("2024-01-04", 1.09),
        ]
    )
    monkeypatch.setattr(httpx, "Client", lambda timeout: _FakeClient(csv))
    out = fetch.fetch_eurusd_ecb(date(2024, 1, 1))
    assert out.columns == ["date", "close"]
    assert out.get_column("close").to_list() == [1.10, 1.11, 1.09]


def test_fetch_eurusd_ecb_filters_target2_holiday_nulls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ECB encodes TARGET-2 holidays as blank OBS_VALUE rows. They must be
    filtered out before gap-validation — otherwise every Easter weekend
    would look like a valid 0-rate observation."""
    csv = _fake_ecb_csv_response(
        [
            ("2024-01-02", 1.10),
            ("2024-01-03", None),  # TARGET-2 holiday
            ("2024-01-04", 1.11),
        ]
    )
    monkeypatch.setattr(httpx, "Client", lambda timeout: _FakeClient(csv))
    out = fetch.fetch_eurusd_ecb(date(2024, 1, 1))
    assert out.height == 2
    assert out.get_column("date").to_list() == [date(2024, 1, 2), date(2024, 1, 4)]


def test_fetch_eurusd_ecb_raises_on_multi_day_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A gap > ECB_FX_MAX_GAP_DAYS (5) in the post-null-filter series
    means a real data anomaly. Loud-fail rather than letting the
    backtest silently forward-fill across the gap (which would bake
    fictitious zero-vol days into every USD-source asset). Mirrors
    the yfinance EURUSD=X mid-2008 incident this fetcher replaces."""
    csv = _fake_ecb_csv_response(
        [
            ("2024-01-02", 1.10),
            ("2024-01-03", 1.11),
            # 8-calendar-day gap — exceeds the 5-day Easter-long-weekend cap
            ("2024-01-11", 1.09),
        ]
    )
    monkeypatch.setattr(httpx, "Client", lambda timeout: _FakeClient(csv))
    with pytest.raises(FetchError, match=r"gap > 5 days"):
        fetch.fetch_eurusd_ecb(date(2024, 1, 1))
