"""Sunday-anchored rebalancing schedules.

Three cadences are supported. A rebalance date is always a Sunday; the signal
uses the preceding Friday's close, and execution is assumed at the following
Monday's close.
"""

from __future__ import annotations

from datetime import date, timedelta

from .universe import Strategy

WEEKLY_SUNDAY = "weekly_sunday"
BIWEEKLY_SUNDAY = "biweekly_sunday"
MONTHLY_FIRST_SUNDAY = "monthly_first_sunday"
SEMIANNUAL_FIRST_SUNDAY = "semiannual_first_sunday"

_SUNDAY = 6  # python weekday()


def is_rebalance_day(strategy: Strategy, day: date) -> bool:
    if day.weekday() != _SUNDAY:
        return False

    if strategy.rebalance == WEEKLY_SUNDAY:
        return True

    if strategy.rebalance == BIWEEKLY_SUNDAY:
        if strategy.reference_date is None:
            raise ValueError(
                f"strategy {strategy.name!r} uses biweekly_sunday but has no reference_date"
            )
        if strategy.reference_date.weekday() != _SUNDAY:
            raise ValueError(
                f"reference_date {strategy.reference_date} for {strategy.name!r} is not a Sunday"
            )
        delta = (day - strategy.reference_date).days
        return delta % 14 == 0

    if strategy.rebalance == MONTHLY_FIRST_SUNDAY:
        return day.day <= 7

    if strategy.rebalance == SEMIANNUAL_FIRST_SUNDAY:
        # First Sunday of January (start of H1) and first Sunday of July
        # (start of H2). Anchored to calendar semesters; if a backtest starts
        # mid-semester the first rebalance is naturally the next semester
        # boundary that falls within the data range.
        return day.month in (1, 7) and day.day <= 7

    raise ValueError(f"Unknown rebalance cadence: {strategy.rebalance!r}")


def rebalance_dates(strategy: Strategy, start: date, end: date) -> list[date]:
    """Return all rebalance Sundays in [start, end], inclusive."""
    if start > end:
        return []
    first_sunday = start + timedelta(days=(_SUNDAY - start.weekday()) % 7)
    out: list[date] = []
    cursor = first_sunday
    while cursor <= end:
        if is_rebalance_day(strategy, cursor):
            out.append(cursor)
        cursor += timedelta(days=7)
    return out


def signal_date(rebalance_day: date) -> date:
    """Friday-before-Sunday: the close used to compute scores."""
    if rebalance_day.weekday() != _SUNDAY:
        raise ValueError(f"{rebalance_day} is not a Sunday")
    return rebalance_day - timedelta(days=2)


def fill_date(rebalance_day: date) -> date:
    """Monday-after-Sunday: the assumed execution close."""
    if rebalance_day.weekday() != _SUNDAY:
        raise ValueError(f"{rebalance_day} is not a Sunday")
    return rebalance_day + timedelta(days=1)
