"""ISO-week calendar in UTC used to bin every weekly quantity.

Weeks run from Monday 00:00 UTC to the next Monday 00:00 UTC. The model has 52 stages per year, so
in ISO years with 53 weeks the last week is folded into week 52, which then lasts 336 hours.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from watervalues.units import DAYS_PER_WEEK, HOURS_PER_WEEK, MODEL_WEEKS_PER_YEAR

CALENDAR_COLUMNS = ("iso_year", "iso_week", "week_start", "week_end", "hours")
_DEC_28 = (12, 28)  # 28 December always falls in the last ISO week of its year


def iso_weeks_in_year(iso_year: int) -> int:
    """Return the number of ISO weeks (52 or 53) in an ISO year."""
    return date(iso_year, *_DEC_28).isocalendar().week


def iso_week_start(iso_year: int, iso_week: int) -> pd.Timestamp:
    """Return the Monday 00:00 UTC that starts an ISO week."""
    return pd.Timestamp(date.fromisocalendar(iso_year, iso_week, 1), tz="UTC").as_unit("ns")


def week_calendar(
    first_iso_year: int, last_iso_year: int, *, fold_week53: bool = True
) -> pd.DataFrame:
    """Build the weekly bins for a range of ISO years.

    Parameters
    ----------
    first_iso_year, last_iso_year : int
        Inclusive range of ISO years.
    fold_week53 : bool
        Merge ISO week 53 into week 52 so that every year has exactly 52 bins.

    Returns
    -------
    pandas.DataFrame
        One row per bin with ``iso_year``, ``iso_week``, ``week_start`` and ``week_end``
        [UTC, half-open interval] and ``hours`` [h]: 168, or 336 for a folded week 52.
    """
    if last_iso_year < first_iso_year:
        raise ValueError("last_iso_year must not be before first_iso_year")
    rows: list[dict[str, object]] = []
    for year in range(first_iso_year, last_iso_year + 1):
        for week in range(1, iso_weeks_in_year(year) + 1):
            start = iso_week_start(year, week)
            end = start + pd.Timedelta(days=DAYS_PER_WEEK)
            if fold_week53 and week > MODEL_WEEKS_PER_YEAR:
                rows[-1]["week_end"] = end
                rows[-1]["hours"] = int(rows[-1]["hours"]) + HOURS_PER_WEEK
                continue
            rows.append(
                {
                    "iso_year": year,
                    "iso_week": week,
                    "week_start": start,
                    "week_end": end,
                    "hours": HOURS_PER_WEEK,
                }
            )
    return pd.DataFrame(rows, columns=list(CALENDAR_COLUMNS))


def assign_bins(timestamps: pd.DatetimeIndex | pd.Series, calendar: pd.DataFrame) -> np.ndarray:
    """Find the calendar bin that contains each timestamp.

    Parameters
    ----------
    timestamps : DatetimeIndex or Series
        Timezone-aware timestamps.
    calendar : pandas.DataFrame
        Output of :func:`week_calendar`.

    Returns
    -------
    numpy.ndarray
        Positional index into ``calendar`` for each timestamp, or -1 when it falls outside.
    """
    ts = pd.DatetimeIndex(timestamps)
    if ts.tz is None:
        raise ValueError("timestamps must be timezone-aware")
    ts = ts.tz_convert("UTC").as_unit("ns")
    starts = pd.DatetimeIndex(calendar["week_start"]).as_unit("ns")
    ends = pd.DatetimeIndex(calendar["week_end"]).as_unit("ns")
    if len(starts) == 0:
        return np.full(len(ts), -1, dtype=np.int64)
    idx = starts.searchsorted(ts, side="right") - 1
    clipped = np.clip(idx, 0, len(starts) - 1)
    inside = (idx >= 0) & np.asarray(ts < ends[clipped])
    return np.where(inside, clipped, -1).astype(np.int64)
