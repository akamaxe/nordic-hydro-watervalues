"""Reservoir storage: from weekly published values to start- and end-of-week storage.

ENTSO-E publishes one value per ISO week (data item 16.1.D) without saying whether it is the
content at the start of the week, at its end, or a weekly mean ("no standard method"). A one-week
misalignment silently corrupts the inflow reconstruction ``I = S_end - S_start + P``, so the
convention is checked against an external reference and against the physics of winter weeks,
where inflow is small and smooth while production varies with demand.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from watervalues.data.calendar import iso_week_start
from watervalues.units import DAYS_PER_WEEK, MWH_PER_GWH

WEEK = pd.Timedelta(days=DAYS_PER_WEEK)
_MIN_WEEKS_FOR_ROUGHNESS = 3  # a standard deviation of week-to-week changes needs >= 3 weeks


class LabelSemantics(StrEnum):
    """What the value published for an ISO week represents."""

    START_OF_WEEK = "start_of_week"
    END_OF_WEEK = "end_of_week"
    WEEKLY_MEAN = "weekly_mean"


@dataclass(frozen=True)
class ReferencePoint:
    """External observation of the storage at the end of an ISO week [GWh]."""

    area: str
    iso_year: int
    iso_week: int
    stored_gwh: float


@dataclass(frozen=True)
class AlignmentEvidence:
    """Result of :func:`diagnose_alignment`."""

    roughness_gwh: dict[str, float]  # median over years of std of week-to-week inflow change
    negative_weeks: dict[str, int]  # winter weeks with negative implied inflow
    best: LabelSemantics
    ratio: float  # roughness of the best hypothesis / second best
    best_by_year: dict[int, str]


def labelled_series(labels: pd.DataFrame) -> pd.Series:
    """Return the published weekly values indexed by the start of their ISO week.

    Parameters
    ----------
    labels : pandas.DataFrame
        Columns ``iso_year``, ``iso_week`` and ``stored_mwh`` [MWh], one row per week.

    Returns
    -------
    pandas.Series
        Values in GWh indexed by the Monday 00:00 UTC that starts each labelled week.
    """
    index = pd.DatetimeIndex(
        [iso_week_start(y, w) for y, w in zip(labels["iso_year"], labels["iso_week"], strict=True)]
    )
    series = pd.Series(labels["stored_mwh"].to_numpy(dtype=float) / MWH_PER_GWH, index=index)
    if series.index.has_duplicates:
        raise ValueError("duplicated reservoir labels: drop revisions first")
    return series.sort_index()


def boundary_snapshots(labels: pd.DataFrame, semantics: LabelSemantics) -> pd.Series:
    """Storage at week boundaries implied by a labelling convention.

    Parameters
    ----------
    labels : pandas.DataFrame
        Columns ``iso_year``, ``iso_week`` and ``stored_mwh`` [MWh].
    semantics : LabelSemantics
        Convention of the published values.

    Returns
    -------
    pandas.Series
        Storage [GWh] indexed by the boundary instant (Monday 00:00 UTC).

    Raises
    ------
    ValueError
        For ``WEEKLY_MEAN``: start-of-week storage is then not observed and the inflow formula
        would have to change, which is a modelling decision.
    """
    if semantics is LabelSemantics.WEEKLY_MEAN:
        raise ValueError(
            "weekly-mean reservoir values do not give start-of-week storage; "
            "the inflow formulation would have to change"
        )
    series = labelled_series(labels)
    if semantics is LabelSemantics.END_OF_WEEK:
        series.index = series.index + WEEK
    return series


def storage_columns(snapshots: pd.Series, calendar: pd.DataFrame) -> pd.DataFrame:
    """Storage at the start and at the end of each calendar bin.

    Parameters
    ----------
    snapshots : pandas.Series
        Output of :func:`boundary_snapshots` [GWh].
    calendar : pandas.DataFrame
        Output of :func:`~watervalues.data.calendar.week_calendar`.

    Returns
    -------
    pandas.DataFrame
        ``storage_gwh`` (start of bin) and ``storage_end_gwh`` (end of bin) [GWh], NaN where no
        snapshot exists. A folded week 52 ends at the start of the next ISO year.
    """
    snaps = snapshots.copy()
    snaps.index = pd.DatetimeIndex(snaps.index).as_unit("ns")
    start = snaps.reindex(pd.DatetimeIndex(calendar["week_start"]).as_unit("ns")).to_numpy()
    end = snaps.reindex(pd.DatetimeIndex(calendar["week_end"]).as_unit("ns")).to_numpy()
    return pd.DataFrame({"storage_gwh": start, "storage_end_gwh": end}, index=calendar.index)


def implied_inflow(
    published_gwh: pd.Series, hydro_gwh: pd.Series, semantics: LabelSemantics
) -> pd.Series:
    """Weekly inflow implied by a labelling convention [GWh].

    Parameters
    ----------
    published_gwh : pandas.Series
        Output of :func:`labelled_series` (consecutive ISO weeks).
    hydro_gwh : pandas.Series
        Hydro production [GWh] indexed by week start.
    semantics : LabelSemantics
        Hypothesis about the published values.

    Returns
    -------
    pandas.Series
        Inflow [GWh] indexed like ``hydro_gwh``; NaN where a neighbouring week is missing.
    """
    x = published_gwh
    t = pd.DatetimeIndex(hydro_gwh.index)
    now = x.reindex(t).to_numpy()
    prev = x.reindex(t - WEEK).to_numpy()
    nxt = x.reindex(t + WEEK).to_numpy()
    p = hydro_gwh.to_numpy(dtype=float)
    if semantics is LabelSemantics.END_OF_WEEK:
        delta = now - prev
    elif semantics is LabelSemantics.START_OF_WEEK:
        delta = nxt - now
    else:  # weekly mean: start ~ (x[w-1] + x[w]) / 2 and end ~ (x[w] + x[w+1]) / 2
        delta = (nxt - prev) / 2
    return pd.Series(delta + p, index=hydro_gwh.index)


def diagnose_alignment(
    published_gwh: pd.Series, hydro_gwh: pd.Series, winter_weeks: tuple[int, int]
) -> AlignmentEvidence:
    """Compare labelling conventions by the smoothness of the implied winter inflow.

    In winter the true inflow is small and slowly varying, while production follows demand.
    Under the right convention the production term cancels; a one-week shift leaves the
    week-to-week production change in the implied inflow (half of it for the weekly mean).

    Parameters
    ----------
    published_gwh : pandas.Series
        Output of :func:`labelled_series`.
    hydro_gwh : pandas.Series
        Hydro production [GWh] per ISO week, indexed by week start.
    winter_weeks : tuple of int
        Inclusive range of ISO weeks used (e.g. ``(2, 14)``).

    Returns
    -------
    AlignmentEvidence
        Roughness and negative-inflow counts per hypothesis, and the winner.
    """
    t = pd.DatetimeIndex(hydro_gwh.index)
    iso = t.isocalendar()
    in_winter = np.asarray((iso["week"] >= winter_weeks[0]) & (iso["week"] <= winter_weeks[1]))
    years = np.asarray(iso["year"])
    roughness: dict[str, float] = {}
    negatives: dict[str, int] = {}
    by_year: dict[str, dict[int, float]] = {}
    for semantics in LabelSemantics:
        inflow = implied_inflow(published_gwh, hydro_gwh, semantics).to_numpy()
        per_year: dict[int, float] = {}
        for year in np.unique(years[in_winter]):
            values = inflow[in_winter & (years == year)]
            values = values[~np.isnan(values)]
            if len(values) >= _MIN_WEEKS_FOR_ROUGHNESS:
                per_year[int(year)] = float(np.std(np.diff(values)))
        by_year[semantics.value] = per_year
        roughness[semantics.value] = float(np.median(list(per_year.values())))
        negatives[semantics.value] = int(np.sum(inflow[in_winter] < 0))
    ranked = sorted(roughness, key=roughness.__getitem__)
    best = LabelSemantics(ranked[0])
    ratio = roughness[ranked[0]] / roughness[ranked[1]]
    common_years = set.intersection(*(set(v) for v in by_year.values()))
    best_by_year = {
        year: min(by_year, key=lambda s, y=year: by_year[s][y]) for year in sorted(common_years)
    }
    return AlignmentEvidence(roughness, negatives, best, ratio, best_by_year)


def reference_residuals(
    published_by_area: Mapping[str, pd.Series], references: Sequence[ReferencePoint]
) -> pd.DataFrame:
    """Difference between published values and external end-of-week references [GWh].

    Under ``END_OF_WEEK`` the reference for week *w* must equal the value labelled *w*; under
    ``START_OF_WEEK`` it must equal the value labelled *w + 1*. A weekly mean matches neither.

    Parameters
    ----------
    published_by_area : mapping of str to pandas.Series
        Output of :func:`labelled_series` per area.
    references : sequence of ReferencePoint
        End-of-week storage observations.

    Returns
    -------
    pandas.DataFrame
        One row per reference with the residual under each convention (NaN if not published).
    """
    rows = []
    for ref in references:
        series = published_by_area[ref.area]
        week_start = iso_week_start(ref.iso_year, ref.iso_week)
        rows.append(
            {
                "area": ref.area,
                "iso_year": ref.iso_year,
                "iso_week": ref.iso_week,
                "reference_gwh": ref.stored_gwh,
                LabelSemantics.END_OF_WEEK.value: series.get(week_start, np.nan) - ref.stored_gwh,
                LabelSemantics.START_OF_WEEK.value: series.get(week_start + WEEK, np.nan)
                - ref.stored_gwh,
            }
        )
    return pd.DataFrame(rows)
