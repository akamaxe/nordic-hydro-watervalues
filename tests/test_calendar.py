"""Tests for the ISO-week calendar."""

import pandas as pd
import pytest

from watervalues.data.calendar import assign_bins, iso_weeks_in_year, week_calendar


def test_iso_years_with_53_weeks() -> None:
    assert [y for y in range(2015, 2027) if iso_weeks_in_year(y) == 53] == [2015, 2020, 2026]


def test_2015_folds_week_53_into_week_52() -> None:
    cal = week_calendar(2015, 2015)
    assert len(cal) == 52
    assert cal["iso_week"].tolist() == list(range(1, 53))
    last = cal.iloc[-1]
    assert last["hours"] == 336
    assert last["week_start"] == pd.Timestamp("2015-12-21", tz="UTC")
    assert last["week_end"] == pd.Timestamp("2016-01-04", tz="UTC")
    assert (cal["hours"].iloc[:-1] == 168).all()


def test_first_week_of_2015_starts_in_december_2014() -> None:
    cal = week_calendar(2015, 2015)
    assert cal.iloc[0]["week_start"] == pd.Timestamp("2014-12-29", tz="UTC")


def test_calendar_is_contiguous_across_years() -> None:
    cal = week_calendar(2015, 2025)
    assert len(cal) == 11 * 52
    assert (cal["week_end"].iloc[:-1].to_numpy() == cal["week_start"].iloc[1:].to_numpy()).all()
    total_hours = (cal["week_end"].iloc[-1] - cal["week_start"].iloc[0]) / pd.Timedelta(hours=1)
    assert cal["hours"].sum() == total_hours


def test_unfolded_calendar_keeps_week_53() -> None:
    cal = week_calendar(2020, 2020, fold_week53=False)
    assert len(cal) == 53
    assert (cal["hours"] == 168).all()


def test_assign_bins_uses_half_open_intervals() -> None:
    cal = week_calendar(2025, 2025)
    ts = pd.DatetimeIndex(
        [
            "2024-12-30 00:00",  # first instant of 2025-W01
            "2025-01-05 23:45",  # last quarter-hour of 2025-W01
            "2025-01-06 00:00",  # first instant of 2025-W02
            "2024-12-29 23:00",  # before the calendar
            "2025-12-29 00:00",  # first instant after the calendar
        ],
        tz="UTC",
    )
    assert assign_bins(ts, cal).tolist() == [0, 0, 1, -1, -1]


def test_assign_bins_converts_local_time_to_utc() -> None:
    cal = week_calendar(2025, 2025)
    # Monday 2025-01-06 00:30 in Stockholm is Sunday 23:30 UTC, still in week 1.
    ts = pd.DatetimeIndex(["2025-01-06 00:30"], tz="Europe/Stockholm")
    assert assign_bins(ts, cal).tolist() == [0]


def test_assign_bins_rejects_naive_timestamps() -> None:
    cal = week_calendar(2025, 2025)
    with pytest.raises(ValueError, match="timezone-aware"):
        assign_bins(pd.DatetimeIndex(["2025-01-01"]), cal)
