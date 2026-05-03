"""Sunday-anchored rebalancing schedules.

Five cadences are supported (weekly / biweekly / monthly / quarterly /
semiannual, plus the buy-and-hold mode that ignores cadence). A rebalance
date is always a Sunday; the signal uses the preceding Friday's close,
and execution is assumed at the following Monday's close.

Cross-cadence start alignment: every cadence rebalances **unconditionally
on the first Sunday of the backtest range**, even when that Sunday isn't
a calendar-anchor day for the cadence's natural rule. Subsequent
rebalances follow the cadence's own rule (calendar-month / quarter /
semester boundary, or biweekly anchor). This guarantees that strategies
with different cadences share the same "day 1" — a backtest starting
mid-quarter has every cadence's first rebalance on the same Sunday,
not 0/1/3/5 months apart depending on cadence.
"""

from __future__ import annotations

from datetime import date, timedelta

from .universe import Strategy

WEEKLY_SUNDAY = "weekly_sunday"
BIWEEKLY_SUNDAY = "biweekly_sunday"
MONTHLY_FIRST_SUNDAY = "monthly_first_sunday"
QUARTERLY_FIRST_SUNDAY = "quarterly_first_sunday"
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

    if strategy.rebalance == QUARTERLY_FIRST_SUNDAY:
        # First Sunday of each calendar quarter (Jan/Apr/Jul/Oct).
        # `rebalance_dates` separately ensures the first Sunday of the
        # backtest range fires unconditionally, so a backtest starting
        # mid-quarter still has its first rebalance on day 1 (then
        # picks up the next calendar-quarter boundary as usual).
        return day.month in (1, 4, 7, 10) and day.day <= 7

    if strategy.rebalance == SEMIANNUAL_FIRST_SUNDAY:
        # First Sunday of January (start of H1) and first Sunday of July
        # (start of H2). Same first-Sunday-of-backtest unconditional-fire
        # guarantee as the other *_first_sunday cadences (see
        # `rebalance_dates`).
        return day.month in (1, 7) and day.day <= 7

    raise ValueError(f"Unknown rebalance cadence: {strategy.rebalance!r}")


def rebalance_dates(strategy: Strategy, start: date, end: date) -> list[date]:
    """Return all rebalance Sundays in [start, end], inclusive.

    Always includes the first Sunday of the backtest range as a rebalance
    day, even when that Sunday isn't a calendar-anchor for the cadence
    (e.g. quarterly_first_sunday with a backtest starting mid-Feb still
    rebalances on the first Sunday of February, not the first Sunday of
    April). This unconditional first-day fire keeps all cadences aligned
    at backtest start so cross-cadence comparisons share the same
    starting allocation.
    """
    if start > end:
        return []
    first_sunday = start + timedelta(days=(_SUNDAY - start.weekday()) % 7)
    out: list[date] = []
    cursor = first_sunday
    while cursor <= end:
        if is_rebalance_day(strategy, cursor):
            out.append(cursor)
        cursor += timedelta(days=7)
    if first_sunday <= end and (not out or out[0] != first_sunday):
        out.insert(0, first_sunday)
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
