"""Time-based aggregation of interval data to hourly and weekly values.

Interval durations always come from the reported resolution, never from row counts: ENTSO-E data
moves from hourly to 15-minute resolution, and both can be present for the same hour.

A normalised interval frame has the columns ``start_utc`` (interval start, timezone-aware),
``resolution_min`` (interval length [min]) and a value column. Optional key columns, such as
``psr_type`` for the production type, identify separate series in the same frame.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from watervalues.data.calendar import assign_bins
from watervalues.units import HOURS_PER_WEEK, MINUTES_PER_HOUR, MWH_PER_GWH

ALLOWED_RESOLUTIONS_MIN = (15, 30, 60)
OTHER_CATEGORY = "other"
_TOL = 1e-9


@dataclass(frozen=True)
class RevisionStats:
    """Rows removed by :func:`drop_revisions`."""

    duplicates: int
    conflicts: int


def drop_revisions(
    df: pd.DataFrame,
    keys: Sequence[str],
    value_columns: Sequence[str],
    update_column: str = "update_utc",
) -> tuple[pd.DataFrame, RevisionStats]:
    """Keep one row per natural key, preferring the most recent update.

    Parameters
    ----------
    df : pandas.DataFrame
        Interval or weekly rows, possibly with several versions of the same key.
    keys : sequence of str
        Columns that identify one observation (e.g. ``start_utc``, ``resolution_min``).
    value_columns : sequence of str
        Columns compared to count conflicting duplicates.
    update_column : str
        Publication time; the latest version wins. Without it, the last row wins.

    Returns
    -------
    tuple of (pandas.DataFrame, RevisionStats)
        Deduplicated rows and the number of duplicates and conflicting duplicates removed.
    """
    keys = list(keys)
    if df.empty:
        return df.copy(), RevisionStats(0, 0)
    order = [*keys, update_column] if update_column in df.columns else keys
    ordered = df.sort_values(order, kind="stable")
    dups = ordered[ordered.duplicated(keys, keep=False)]
    conflicts = 0
    if not dups.empty:
        n_unique = dups.groupby(keys)[list(value_columns)].nunique(dropna=False)
        conflicts = int((n_unique > 1).any(axis=1).sum())
    kept = ordered.drop_duplicates(keys, keep="last").reset_index(drop=True)
    return kept, RevisionStats(duplicates=len(ordered) - len(kept), conflicts=conflicts)


def _hourly_parts(df: pd.DataFrame, value_column: str, by: Sequence[str]) -> pd.DataFrame:
    """Integrate intervals into hours, keeping one resolution per series and hour."""
    by = list(by)
    resolution = df["resolution_min"].to_numpy(dtype=np.int64)
    unsupported = sorted(set(resolution[~np.isin(resolution, ALLOWED_RESOLUTIONS_MIN)]))
    if unsupported:
        raise ValueError(f"unsupported resolutions [min]: {unsupported}")
    start = pd.DatetimeIndex(df["start_utc"])
    if start.tz is None:
        raise ValueError("start_utc must be timezone-aware")
    start = start.tz_convert("UTC").as_unit("ns")
    hour = start.floor("h")
    minute = np.asarray((start - hour) / pd.Timedelta(minutes=1))
    if np.any(minute % resolution != 0):
        raise ValueError("interval starts are not aligned to their resolution")
    duration_h = resolution / MINUTES_PER_HOUR
    values = df[value_column].to_numpy(dtype=float)
    ok = ~np.isnan(values)
    work = pd.DataFrame(
        {
            **{col: df[col].to_numpy()[ok] for col in by},
            "hour_utc": hour[ok],
            "resolution_min": resolution[ok],
            "weighted": values[ok] * duration_h[ok],
            "covered_h": duration_h[ok],
        }
    )
    keys = [*by, "hour_utc"]
    parts = work.groupby([*keys, "resolution_min"], sort=False).sum().reset_index()
    if (parts["covered_h"] > 1 + _TOL).any():
        raise ValueError("more than one hour of data in an hour: drop duplicated intervals first")
    # Prefer the resolution that covers most of the hour; on ties, the finer one.
    parts = parts.sort_values(
        [*keys, "covered_h", "resolution_min"],
        ascending=[True] * len(keys) + [False, True],
        kind="stable",
    )
    best = parts.drop_duplicates(keys, keep="first").drop(columns="resolution_min")
    return best.reset_index(drop=True)


def hourly_energy(
    df: pd.DataFrame, value_column: str = "value_mw", by: Sequence[str] = ()
) -> pd.DataFrame:
    """Integrate average power per interval into energy per hour.

    Parameters
    ----------
    df : pandas.DataFrame
        Normalised interval frame with average power [MW] in ``value_column``.
    value_column : str
        Column holding power [MW].
    by : sequence of str
        Key columns that identify separate series (e.g. ``("psr_type",)``).

    Returns
    -------
    pandas.DataFrame
        ``*by``, ``hour_utc``, ``energy_mwh`` [MWh] and ``covered_h`` [h, 0–1].
    """
    return _hourly_parts(df, value_column, by).rename(columns={"weighted": "energy_mwh"})


def hourly_mean(
    df: pd.DataFrame, value_column: str = "price_eur_mwh", by: Sequence[str] = ()
) -> pd.DataFrame:
    """Time-weighted hourly mean of an intensive quantity such as a price.

    Parameters
    ----------
    df : pandas.DataFrame
        Normalised interval frame.
    value_column : str
        Column to average (e.g. price [EUR/MWh]).
    by : sequence of str
        Key columns that identify separate series.

    Returns
    -------
    pandas.DataFrame
        ``*by``, ``hour_utc``, ``value_column`` (hourly mean) and ``covered_h`` [h, 0–1].
    """
    parts = _hourly_parts(df, value_column, by)
    parts[value_column] = parts["weighted"] / parts["covered_h"]
    return parts.drop(columns="weighted")


def _bin_sums(
    hourly: pd.DataFrame, calendar: pd.DataFrame, by: Sequence[str], columns: Sequence[str]
) -> pd.DataFrame:
    """Sum hourly columns per calendar bin, with a row for every bin and series."""
    by = list(by)
    bins = assign_bins(hourly["hour_utc"], calendar)
    inside = bins >= 0
    sums = (
        hourly.loc[inside, [*by, *columns]]
        .assign(bin=bins[inside])
        .groupby([*by, "bin"], sort=True)[list(columns)]
        .sum()
    )
    all_bins = range(len(calendar))
    if by:
        groups = hourly[by].drop_duplicates().itertuples(index=False, name=None)
        full = pd.MultiIndex.from_tuples(
            [(*group, b) for group in groups for b in all_bins], names=[*by, "bin"]
        )
    else:
        full = pd.Index(all_bins, name="bin")
    return sums.reindex(full, fill_value=0.0)


def _coverage(sums: pd.DataFrame, calendar: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Covered hours and the bin length [h] for each row of a ``_bin_sums`` result."""
    positions = sums.index.get_level_values("bin").to_numpy()
    hours = calendar["hours"].to_numpy(dtype=float)[positions]
    return sums["covered_h"].to_numpy(dtype=float), hours


def weekly_energy(
    hourly: pd.DataFrame,
    calendar: pd.DataFrame,
    min_coverage: float,
    by: Sequence[str] = (),
) -> pd.DataFrame:
    """Aggregate hourly energy to calendar bins.

    Bins covered at least ``min_coverage`` are scaled by ``hours / covered hours``, which fills
    the gaps with the bin's mean power; less covered bins are NaN.

    Parameters
    ----------
    hourly : pandas.DataFrame
        Output of :func:`hourly_energy`.
    calendar : pandas.DataFrame
        Output of :func:`~watervalues.data.calendar.week_calendar`.
    min_coverage : float
        Minimum share of the bin's hours with data (0–1].
    by : sequence of str
        Key columns that identify separate series.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``(*by, bin)`` with ``energy_gwh`` [GWh] and ``coverage`` [-].
    """
    sums = _bin_sums(hourly, calendar, by, ["energy_mwh", "covered_h"])
    covered, hours = _coverage(sums, calendar)
    coverage = covered / hours
    energy = np.full(len(sums), np.nan)
    enough = coverage >= min_coverage
    energy[enough] = (
        sums["energy_mwh"].to_numpy()[enough] * hours[enough] / covered[enough] / MWH_PER_GWH
    )
    return pd.DataFrame({"energy_gwh": energy, "coverage": coverage}, index=sums.index)


def weekly_mean(
    hourly: pd.DataFrame,
    calendar: pd.DataFrame,
    min_coverage: float,
    value_column: str = "price_eur_mwh",
    by: Sequence[str] = (),
) -> pd.DataFrame:
    """Aggregate an hourly intensive quantity to a time-weighted mean per calendar bin.

    Parameters
    ----------
    hourly : pandas.DataFrame
        Output of :func:`hourly_mean`.
    calendar : pandas.DataFrame
        Output of :func:`~watervalues.data.calendar.week_calendar`.
    min_coverage : float
        Minimum share of the bin's hours with data (0–1].
    value_column : str
        Column to average (e.g. price [EUR/MWh]).
    by : sequence of str
        Key columns that identify separate series.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``(*by, bin)`` with ``value_column`` (NaN below ``min_coverage``) and
        ``coverage`` [-].
    """
    weighted = hourly.assign(weighted=hourly[value_column] * hourly["covered_h"])
    sums = _bin_sums(weighted, calendar, by, ["weighted", "covered_h"])
    covered, hours = _coverage(sums, calendar)
    coverage = covered / hours
    mean = np.full(len(sums), np.nan)
    enough = coverage >= min_coverage
    mean[enough] = sums["weighted"].to_numpy()[enough] / covered[enough]
    return pd.DataFrame({value_column: mean, "coverage": coverage}, index=sums.index)


def weekly_load_weighted_price(
    prices: Mapping[str, pd.DataFrame],
    loads: Mapping[str, pd.DataFrame],
    calendar: pd.DataFrame,
    min_coverage: float,
) -> pd.DataFrame:
    """Load-weighted weekly price over several bidding zones.

    Only hours in which every zone has both a price and a load are used, so a missing zone never
    biases the average towards the others.

    Parameters
    ----------
    prices : mapping of str to pandas.DataFrame
        Per zone, the output of :func:`hourly_mean` with ``price_eur_mwh`` [EUR/MWh].
    loads : mapping of str to pandas.DataFrame
        Per zone, the output of :func:`hourly_energy` for the load [MWh per hour].
    calendar : pandas.DataFrame
        Output of :func:`~watervalues.data.calendar.week_calendar`.
    min_coverage : float
        Minimum share of the bin's hours with complete data (0–1].

    Returns
    -------
    pandas.DataFrame
        Indexed by ``bin`` with ``price_eur_mwh`` [EUR/MWh] and ``coverage`` [-].
    """
    if set(prices) != set(loads):
        raise ValueError("prices and loads must cover the same zones")
    zones = sorted(prices)
    price = pd.DataFrame({z: prices[z].set_index("hour_utc")["price_eur_mwh"] for z in zones})
    load_mw = pd.DataFrame(
        {z: loads[z].set_index("hour_utc").eval("energy_mwh / covered_h") for z in zones}
    )
    price, load_mw = price.align(load_mw, join="outer")
    complete = price.notna().all(axis=1) & load_mw.notna().all(axis=1)
    hourly = pd.DataFrame(
        {
            "hour_utc": price.index[complete],
            "numerator": (price * load_mw).sum(axis=1)[complete].to_numpy(),
            "denominator": load_mw.sum(axis=1)[complete].to_numpy(),
            "covered_h": 1.0,
        }
    )
    sums = _bin_sums(hourly, calendar, (), ["numerator", "denominator", "covered_h"])
    covered, hours = _coverage(sums, calendar)
    coverage = covered / hours
    mean = np.full(len(sums), np.nan)
    enough = coverage >= min_coverage
    mean[enough] = sums["numerator"].to_numpy()[enough] / sums["denominator"].to_numpy()[enough]
    return pd.DataFrame({"price_eur_mwh": mean, "coverage": coverage}, index=sums.index)


def stitch(
    segments: Sequence[tuple[pd.DataFrame, pd.Timestamp | None, pd.Timestamp | None]],
    time_column: str = "hour_utc",
) -> pd.DataFrame:
    """Concatenate hourly frames, each restricted to its half-open validity period.

    Parameters
    ----------
    segments : sequence of (frame, start, end)
        Each frame is kept for ``start <= time < end``; ``None`` leaves that side open.
    time_column : str
        Column with timezone-aware timestamps.

    Returns
    -------
    pandas.DataFrame
        The concatenated rows, sorted by time.
    """
    parts = []
    for frame, start, end in segments:
        keep = np.ones(len(frame), dtype=bool)
        if start is not None:
            keep &= np.asarray(frame[time_column] >= start)
        if end is not None:
            keep &= np.asarray(frame[time_column] < end)
        parts.append(frame[keep])
    out = pd.concat(parts, ignore_index=True).sort_values(time_column, kind="stable")
    if out[time_column].duplicated().any():
        raise ValueError("segments overlap in time")
    return out.reset_index(drop=True)


def weekly_generation(
    hourly_by_psr: pd.DataFrame,
    calendar: pd.DataFrame,
    categories: Mapping[str, Sequence[str]],
    known_psr_types: Sequence[str],
    min_coverage: float,
    partial_period_types: Sequence[str] = (),
) -> pd.DataFrame:
    """Weekly generation per technology category.

    A production type listed in ``partial_period_types`` that is absent for a whole bin counts
    as zero when the area reported other types in that bin and the bin lies outside the type's
    reporting period (for example solar before its first report). Any other gap follows the
    coverage rule, so a type that is not listed and stops early (say, nuclear missing in the last
    week of the cache) is missing, not zero. A category is NaN when any of its types is NaN.

    Parameters
    ----------
    hourly_by_psr : pandas.DataFrame
        Output of :func:`hourly_energy` with ``by=("psr_type",)``.
    calendar : pandas.DataFrame
        Output of :func:`~watervalues.data.calendar.week_calendar`.
    categories : mapping of str to sequence of str
        Category name to PSR codes; codes not listed go to ``"other"``.
    known_psr_types : sequence of str
        Every valid PSR code; an unknown code in the data is an error.
    min_coverage : float
        Minimum share of the bin's hours with data (0–1].
    partial_period_types : sequence of str
        PSR codes reported only during part of the period, which count as zero outside it.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``bin`` with one ``<category>_gwh`` column [GWh] per category plus
        ``other_gwh``.
    """
    present = sorted(hourly_by_psr["psr_type"].unique())
    unknown = sorted(set(present) - set(known_psr_types))
    if unknown:
        raise ValueError(f"unknown production types in the data: {unknown}")
    category_of = {code: name for name, codes in categories.items() for code in codes}
    weekly = weekly_energy(hourly_by_psr, calendar, min_coverage, by=("psr_type",))

    psr = weekly.index.get_level_values("psr_type")
    positions = weekly.index.get_level_values("bin").to_numpy()
    area_coverage = weekly["coverage"].groupby(level="bin").max()
    area_reported = area_coverage.reindex(positions).to_numpy() >= min_coverage

    first_hour = pd.DatetimeIndex(hourly_by_psr.groupby("psr_type")["hour_utc"].min().reindex(psr))
    last_hour = pd.DatetimeIndex(hourly_by_psr.groupby("psr_type")["hour_utc"].max().reindex(psr))
    week_start = pd.DatetimeIndex(calendar["week_start"])[positions]
    week_end = pd.DatetimeIndex(calendar["week_end"])[positions]
    outside_span = np.asarray((week_end <= first_hour) | (week_start > last_hour))
    partial = np.isin(psr, list(partial_period_types))
    energy = weekly["energy_gwh"].to_numpy().copy()
    structural_zero = (weekly["coverage"].to_numpy() == 0) & area_reported & outside_span & partial
    energy[structural_zero] = 0.0

    category = [category_of.get(code, OTHER_CATEGORY) for code in psr]
    frame = pd.DataFrame({"energy_gwh": energy, "category": category, "bin": positions})
    totals = frame.groupby(["category", "bin"])["energy_gwh"].sum()
    has_nan = frame["energy_gwh"].isna().groupby([frame["category"], frame["bin"]]).any()
    totals[has_nan] = np.nan
    wide = totals.unstack("category").reindex(range(len(calendar)))
    reported = area_coverage.reindex(range(len(calendar))).to_numpy() >= min_coverage
    out = pd.DataFrame(index=pd.Index(range(len(calendar)), name="bin"))
    for name in [*categories, OTHER_CATEGORY]:
        if name in wide.columns:
            out[f"{name}_gwh"] = wide[name].to_numpy()
        else:
            # No production type of this category in the data: zero where the area reported.
            out[f"{name}_gwh"] = np.where(reported, 0.0, np.nan)
    return out


def per_stage(
    weekly: pd.DataFrame,
    energy_columns: Sequence[str],
    stage_hours: float = HOURS_PER_WEEK,
) -> pd.DataFrame:
    """Normalise weekly energies to a standard stage length.

    A folded week 52 (``hours = 336``) becomes an average 168-hour week, consistent with the
    weekly bounds of the SDP. Storage levels and prices are left unchanged, so the storage change
    of a folded row still spans 336 hours: compute flows that come from the energy balance
    (``inflow = storage_end - storage + hydro``) on the unscaled rows first and pass them here
    with the other energies. Computing them after this call breaks the balance.

    Parameters
    ----------
    weekly : pandas.DataFrame
        Weekly dataset with an ``hours`` column [h].
    energy_columns : sequence of str
        Energy columns [GWh] to rescale.
    stage_hours : float
        Target stage length [h].

    Returns
    -------
    pandas.DataFrame
        A copy with the energy columns scaled by ``stage_hours / hours``.
    """
    out = weekly.copy()
    factor = stage_hours / weekly["hours"].astype(float)
    for column in energy_columns:
        out[column] = weekly[column] * factor
    return out
