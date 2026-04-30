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
        description="",
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
