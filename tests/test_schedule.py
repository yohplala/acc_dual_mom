"""Tests for Sunday-anchored rebalancing schedules."""

from __future__ import annotations

from datetime import date

import pytest

from pea_momentum.schedule import (
    fill_date,
    is_rebalance_day,
    rebalance_dates,
    signal_date,
)
from pea_momentum.universe import Strategy


def _strategy(rebalance: str, reference_date: date | None = None) -> Strategy:
    return Strategy(
        name="t",
        asset_ids=("a",),
        rebalance=rebalance,
        top_n=1,
        reference_date=reference_date,
    )


class TestWeeklySunday:
    def test_sundays_match(self) -> None:
        s = _strategy("weekly_sunday")
        assert is_rebalance_day(s, date(2026, 4, 26))  # a Sunday
        assert is_rebalance_day(s, date(2026, 5, 3))

    def test_non_sundays_reject(self) -> None:
        s = _strategy("weekly_sunday")
        for offset in range(1, 7):
            day = date(2026, 4, 26) + __import__("datetime").timedelta(days=offset - 7)
            if day.weekday() != 6:
                assert not is_rebalance_day(s, day)


class TestBiweeklySunday:
    def test_anchored_sundays(self) -> None:
        s = _strategy("biweekly_sunday", reference_date=date(2020, 1, 5))
        assert is_rebalance_day(s, date(2020, 1, 5))
        assert is_rebalance_day(s, date(2020, 1, 19))
        assert not is_rebalance_day(s, date(2020, 1, 12))

    def test_works_far_from_anchor(self) -> None:
        s = _strategy("biweekly_sunday", reference_date=date(2020, 1, 5))
        assert is_rebalance_day(s, date(2026, 4, 19))
        assert is_rebalance_day(s, date(2026, 5, 3))
        assert not is_rebalance_day(s, date(2026, 4, 26))

    def test_missing_anchor_raises(self) -> None:
        s = _strategy("biweekly_sunday", reference_date=None)
        with pytest.raises(ValueError, match="reference_date"):
            is_rebalance_day(s, date(2026, 4, 26))

    def test_non_sunday_anchor_raises(self) -> None:
        s = _strategy("biweekly_sunday", reference_date=date(2020, 1, 6))
        with pytest.raises(ValueError, match="not a Sunday"):
            is_rebalance_day(s, date(2026, 4, 26))


class TestMonthlyFirstSunday:
    @pytest.mark.parametrize(
        "day",
        [
            date(2026, 1, 4),
            date(2026, 2, 1),
            date(2026, 3, 1),
            date(2026, 4, 5),
            date(2026, 5, 3),
            date(2026, 6, 7),
        ],
    )
    def test_first_sundays_match(self, day: date) -> None:
        s = _strategy("monthly_first_sunday")
        assert is_rebalance_day(s, day)

    @pytest.mark.parametrize(
        "day",
        [
            date(2026, 4, 12),
            date(2026, 4, 19),
            date(2026, 4, 26),
            date(2026, 5, 10),
        ],
    )
    def test_later_sundays_reject(self, day: date) -> None:
        s = _strategy("monthly_first_sunday")
        assert not is_rebalance_day(s, day)


class TestQuarterlyFirstSunday:
    @pytest.mark.parametrize(
        "day",
        [
            date(2026, 1, 4),  # Q1
            date(2026, 4, 5),  # Q2
            date(2026, 7, 5),  # Q3
            date(2026, 10, 4),  # Q4
            date(2024, 1, 7),
            date(2024, 4, 7),
            date(2024, 7, 7),
            date(2024, 10, 6),
        ],
    )
    def test_first_sundays_of_each_quarter_match(self, day: date) -> None:
        s = _strategy("quarterly_first_sunday")
        assert is_rebalance_day(s, day)

    @pytest.mark.parametrize(
        "day",
        [
            date(2026, 2, 1),  # Feb (not a quarter-start month)
            date(2026, 5, 3),  # May
            date(2026, 8, 2),  # Aug
            date(2026, 11, 1),  # Nov
            date(2026, 1, 11),  # second Sunday of January
            date(2026, 4, 12),  # second Sunday of April
            date(2026, 12, 6),  # December
        ],
    )
    def test_other_sundays_reject(self, day: date) -> None:
        s = _strategy("quarterly_first_sunday")
        assert not is_rebalance_day(s, day)


class TestSemiannualFirstSunday:
    @pytest.mark.parametrize(
        "day",
        [
            date(2026, 1, 4),  # first Sunday of January
            date(2026, 7, 5),  # first Sunday of July
            date(2024, 1, 7),
            date(2024, 7, 7),
        ],
    )
    def test_first_sundays_of_h1_h2_match(self, day: date) -> None:
        s = _strategy("semiannual_first_sunday")
        assert is_rebalance_day(s, day)

    @pytest.mark.parametrize(
        "day",
        [
            date(2026, 2, 1),  # first Sunday of Feb — wrong month
            date(2026, 6, 7),  # first Sunday of June
            date(2026, 8, 2),  # first Sunday of August
            date(2026, 1, 11),  # second Sunday of January
            date(2026, 7, 12),  # second Sunday of July
            date(2026, 12, 6),  # first Sunday of December
        ],
    )
    def test_other_sundays_reject(self, day: date) -> None:
        s = _strategy("semiannual_first_sunday")
        assert not is_rebalance_day(s, day)


class TestRebalanceDates:
    def test_weekly_yields_all_sundays(self) -> None:
        s = _strategy("weekly_sunday")
        dates = rebalance_dates(s, date(2026, 4, 1), date(2026, 4, 30))
        assert dates == [date(2026, 4, 5), date(2026, 4, 12), date(2026, 4, 19), date(2026, 4, 26)]

    def test_monthly_yields_first_sundays(self) -> None:
        s = _strategy("monthly_first_sunday")
        dates = rebalance_dates(s, date(2026, 1, 1), date(2026, 6, 30))
        assert dates == [
            date(2026, 1, 4),
            date(2026, 2, 1),
            date(2026, 3, 1),
            date(2026, 4, 5),
            date(2026, 5, 3),
            date(2026, 6, 7),
        ]

    def test_empty_when_start_after_end(self) -> None:
        s = _strategy("weekly_sunday")
        assert rebalance_dates(s, date(2026, 5, 1), date(2026, 4, 1)) == []

    def test_quarterly_yields_four_sundays_per_year(self) -> None:
        s = _strategy("quarterly_first_sunday")
        dates = rebalance_dates(s, date(2024, 1, 1), date(2024, 12, 31))
        assert dates == [
            date(2024, 1, 7),
            date(2024, 4, 7),
            date(2024, 7, 7),
            date(2024, 10, 6),
        ]

    def test_quarterly_mid_quarter_start_includes_first_sunday_of_backtest(self) -> None:
        # Backtest starting mid-Q1 (Feb 1): first rebalance unconditionally
        # fires on the first Sunday of the backtest range (Feb 4), then
        # subsequent rebalances pick up at the next calendar-quarter
        # boundaries (Q2/Q3/Q4 first Sundays).
        s = _strategy("quarterly_first_sunday")
        dates = rebalance_dates(s, date(2024, 2, 1), date(2024, 12, 31))
        assert dates == [
            date(2024, 2, 4),
            date(2024, 4, 7),
            date(2024, 7, 7),
            date(2024, 10, 6),
        ]

    def test_semiannual_yields_two_sundays_per_year(self) -> None:
        s = _strategy("semiannual_first_sunday")
        dates = rebalance_dates(s, date(2024, 1, 1), date(2025, 12, 31))
        assert dates == [
            date(2024, 1, 7),
            date(2024, 7, 7),
            date(2025, 1, 5),
            date(2025, 7, 6),
        ]

    def test_semiannual_mid_h1_start_includes_first_sunday_of_backtest(self) -> None:
        # Backtest starting in February: first rebalance fires
        # unconditionally on the first Sunday of backtest (Feb 4), then
        # the next semester boundary (Jul 7).
        s = _strategy("semiannual_first_sunday")
        dates = rebalance_dates(s, date(2024, 2, 1), date(2024, 12, 31))
        assert dates == [date(2024, 2, 4), date(2024, 7, 7)]

    @pytest.mark.parametrize(
        "cadence",
        [
            "weekly_sunday",
            "monthly_first_sunday",
            "quarterly_first_sunday",
            "semiannual_first_sunday",
        ],
    )
    def test_first_sunday_of_backtest_always_fires_regardless_of_cadence(
        self, cadence: str
    ) -> None:
        # Cross-cadence guarantee: every cadence has its first rebalance
        # on the first Sunday of the backtest range (here, 2024-02-04 for
        # a backtest starting Feb 1, 2024). This keeps strategies aligned
        # at "day 1" so cross-cadence comparisons share the same starting
        # allocation.
        s = _strategy(cadence)
        dates = rebalance_dates(s, date(2024, 2, 1), date(2024, 12, 31))
        assert dates[0] == date(2024, 2, 4)


class TestSignalAndFillDates:
    def test_signal_is_friday_before(self) -> None:
        sunday = date(2026, 4, 26)
        assert signal_date(sunday) == date(2026, 4, 24)
        assert signal_date(sunday).weekday() == 4

    def test_fill_is_monday_after(self) -> None:
        sunday = date(2026, 4, 26)
        assert fill_date(sunday) == date(2026, 4, 27)
        assert fill_date(sunday).weekday() == 0

    def test_non_sunday_raises(self) -> None:
        with pytest.raises(ValueError, match="not a Sunday"):
            signal_date(date(2026, 4, 25))
        with pytest.raises(ValueError, match="not a Sunday"):
            fill_date(date(2026, 4, 25))
