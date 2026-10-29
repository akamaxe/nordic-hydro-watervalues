"""Build the weekly dataset ``data/processed/weekly_se.csv`` from the normalised raw cache.

The build also checks the reservoir labelling convention against the external references and
the physics of winter weeks, and prints a sanity report. A failed convention check stops the
build: a one-week misalignment would silently corrupt every inflow downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from watervalues.config import PROJECT_ROOT, AppConfig
from watervalues.data.aggregate import (
    hourly_energy,
    hourly_mean,
    per_stage,
    stitch,
    weekly_energy,
    weekly_generation,
    weekly_load_weighted_price,
    weekly_mean,
)
from watervalues.data.cache import read_dataset
from watervalues.data.calendar import week_calendar
from watervalues.data.filelib import Manifest
from watervalues.data.reservoir import (
    AlignmentEvidence,
    LabelSemantics,
    ReferencePoint,
    boundary_snapshots,
    diagnose_alignment,
    fill_short_gaps,
    labelled_series,
    reference_residuals,
    storage_columns,
)
from watervalues.data.schema import (
    KNOWN_PSR_TYPES,
    PUMPED_STORAGE_PSR,
    WEEKLY_COLUMNS,
)
from watervalues.units import HOURS_PER_WEEK, MWH_PER_GWH

GWH_PER_TWH = MWH_PER_GWH  # 1 TWh = 1000 GWh


class BuildError(RuntimeError):
    """A consistency check failed; the weekly dataset must not be used."""


@dataclass
class BuildResult:
    """Weekly dataset plus everything needed to judge it."""

    weekly: pd.DataFrame
    report: list[str]
    references: pd.DataFrame
    evidence: AlignmentEvidence
    interpolated_storage: list[pd.Timestamp]


def week_label(iso_year: int, iso_week: int) -> str:
    """Return a label such as ``2015-W01``."""
    return f"{iso_year}-W{iso_week:02d}"


def _zone_column(zone: str) -> str:
    return f"price_{zone.lower()}_eur_mwh"


def check_reservoir_convention(
    cfg: AppConfig, source: str, hydro_by_week: pd.Series
) -> tuple[pd.DataFrame, AlignmentEvidence]:
    """Verify the configured reservoir convention against references and winter physics.

    Parameters
    ----------
    cfg : AppConfig
        Configuration with ``data.reservoir``.
    source : str
        Raw-cache source to read.
    hydro_by_week : pandas.Series
        Hydro production [GWh] per ISO week (unfolded), indexed by week start.

    Returns
    -------
    tuple of (pandas.DataFrame, AlignmentEvidence)
        Reference residuals and the physics diagnosis.

    Raises
    ------
    BuildError
        If either check contradicts the configured convention.
    """
    res = cfg.data.reservoir
    raw_dir = PROJECT_ROOT / cfg.data.raw_dir
    configured = LabelSemantics(res.label_semantics)
    published = {
        area: labelled_series(read_dataset(raw_dir, source, "reservoir", area))
        for area in sorted({res.area, *(ref.area for ref in res.references)})
    }
    references = reference_residuals(
        published,
        [ReferencePoint(r.area, r.iso_year, r.iso_week, r.stored_gwh) for r in res.references],
    )
    if not references.empty:
        residual = references[configured.value].dropna().abs()
        if (residual > res.reference_tolerance_gwh).any():
            raise BuildError(
                f"reservoir values contradict the external references under {configured.value}:\n"
                f"{references.to_string(index=False)}"
            )
    evidence = diagnose_alignment(published[res.area], hydro_by_week, tuple(res.winter_weeks))
    if evidence.best is not configured or evidence.ratio > res.max_roughness_ratio:
        raise BuildError(
            f"winter physics favour {evidence.best.value} (roughness ratio {evidence.ratio:.2f}), "
            f"configured {configured.value}: {evidence.roughness_gwh}"
        )
    return references, evidence


def build_weekly(cfg: AppConfig, source: str = "filelib") -> BuildResult:
    """Assemble the weekly dataset from the raw cache.

    Parameters
    ----------
    cfg : AppConfig
        Model configuration.
    source : str
        Raw-cache source (``"filelib"`` or ``"entsoe_api"``).

    Returns
    -------
    BuildResult
        The dataset in the column order of ``WEEKLY_COLUMNS`` and its sanity report.
    """
    data = cfg.data
    raw_dir = PROJECT_ROOT / data.raw_dir
    first, last = cfg.period.first_iso_year, cfg.period.last_iso_year
    calendar = week_calendar(first, last)
    coverage = data.min_coverage

    def read(dataset: str, area: str) -> pd.DataFrame:
        return read_dataset(raw_dir, source, dataset, area)

    generation = read("generation", data.country)
    hourly_generation = hourly_energy(generation, "value_mw", by=("psr_type",))
    categories = data.generation_categories
    by_category = weekly_generation(
        hourly_generation, calendar, categories, KNOWN_PSR_TYPES, coverage
    )
    pumping_rows = generation[generation["psr_type"] == PUMPED_STORAGE_PSR]
    pumping_gwh = (
        weekly_energy(hourly_energy(pumping_rows, "consumption_mw"), calendar, coverage)[
            "energy_gwh"
        ].to_numpy()
        if len(pumping_rows)
        else np.zeros(len(calendar))
    )

    load_gwh = weekly_energy(hourly_energy(read("load", data.country)), calendar, coverage)
    zone_load = {z: hourly_energy(read("load", z)) for z in data.price_zones}
    zone_price = {z: hourly_mean(read("price", z), "price_eur_mwh") for z in data.price_zones}
    price_se = weekly_load_weighted_price(zone_price, zone_load, calendar, coverage)
    switch = pd.Timestamp(data.de_price.switch_utc).tz_convert("UTC")
    de_hourly = stitch(
        [
            (hourly_mean(read("price", data.de_price.before), "price_eur_mwh"), None, switch),
            (hourly_mean(read("price", data.de_price.after), "price_eur_mwh"), switch, None),
        ]
    )

    semantics = LabelSemantics(data.reservoir.label_semantics)
    labels = read("reservoir", data.reservoir.area)
    snapshots, interpolated = fill_short_gaps(
        boundary_snapshots(labels, semantics), data.reservoir.max_interp_weeks
    )
    storage = storage_columns(snapshots, calendar)

    unfolded = week_calendar(first, last, fold_week53=False)
    hydro_unfolded = weekly_generation(
        hourly_generation, unfolded, categories, KNOWN_PSR_TYPES, coverage
    )["hydro_gwh"]
    hydro_by_week = pd.Series(
        hydro_unfolded.to_numpy(), index=pd.DatetimeIndex(unfolded["week_start"])
    )
    references, evidence = check_reservoir_convention(cfg, source, hydro_by_week)

    columns = {
        "week_start": calendar["week_start"].to_numpy(),
        "iso_year": calendar["iso_year"].to_numpy(),
        "iso_week": calendar["iso_week"].to_numpy(),
        "hours": calendar["hours"].to_numpy(),
        "storage_gwh": storage["storage_gwh"].to_numpy(),
        "storage_end_gwh": storage["storage_end_gwh"].to_numpy(),
        "hydro_gwh": by_category["hydro_gwh"].to_numpy(),
        "load_gwh": load_gwh["energy_gwh"].to_numpy(),
        "nuclear_gwh": by_category["nuclear_gwh"].to_numpy(),
        "wind_gwh": by_category["wind_gwh"].to_numpy(),
        "solar_gwh": by_category["solar_gwh"].to_numpy(),
        "other_gwh": by_category["other_gwh"].to_numpy(),
        "price_se_eur_mwh": price_se["price_eur_mwh"].to_numpy(),
    }
    for zone in data.price_zones:
        columns[_zone_column(zone)] = weekly_mean(zone_price[zone], calendar, coverage)[
            "price_eur_mwh"
        ].to_numpy()
    columns["price_de_eur_mwh"] = weekly_mean(de_hourly, calendar, coverage)[
        "price_eur_mwh"
    ].to_numpy()
    weekly = pd.DataFrame(columns)
    if tuple(weekly.columns) != WEEKLY_COLUMNS:
        raise BuildError(f"weekly columns {tuple(weekly.columns)} differ from the contract")

    zone_load_gwh = sum(
        weekly_energy(zone_load[z], calendar, coverage)["energy_gwh"].to_numpy()
        for z in data.price_zones
    )
    report = sanity_report(
        cfg, weekly, pumping_gwh, zone_load_gwh, references, evidence, interpolated, raw_dir
    )
    return BuildResult(weekly, report, references, evidence, interpolated)


def sanity_report(
    cfg: AppConfig,
    weekly: pd.DataFrame,
    pumping_gwh: np.ndarray,
    zone_load_gwh: np.ndarray,
    references: pd.DataFrame,
    evidence: AlignmentEvidence,
    interpolated: list[pd.Timestamp],
    raw_dir: Path,
) -> list[str]:
    """Human-readable checks of the weekly dataset (orders of magnitude, consistency, gaps)."""
    sanity = cfg.data.sanity
    w = weekly.copy()
    w["thermal_and_res_gwh"] = w[["nuclear_gwh", "wind_gwh", "solar_gwh", "other_gwh"]].sum(
        axis=1, min_count=4
    )
    w["residual_demand_gwh"] = w["load_gwh"] - w["thermal_and_res_gwh"]
    w["net_import_gwh"] = w["residual_demand_gwh"] - w["hydro_gwh"]
    w["inflow_gwh"] = w["storage_end_gwh"] - w["storage_gwh"] + w["hydro_gwh"]
    w["pumping_gwh"] = pumping_gwh
    w["zone_load_gwh"] = zone_load_gwh
    lines = ["Annual totals per ISO year [TWh]; storage [GWh]; hour-weighted mean prices [EUR/MWh]"]
    rows = []
    for year, g in w.groupby("iso_year"):

        def twh(column: str, frame: pd.DataFrame = g) -> float:
            return frame[column].sum(min_count=1) / GWH_PER_TWH

        def mean_price(column: str, frame: pd.DataFrame = g) -> float:
            ok = frame[column].notna()
            return float(np.average(frame.loc[ok, column], weights=frame.loc[ok, "hours"]))

        rows.append(
            {
                "year": year,
                "load": twh("load_gwh"),
                "hydro": twh("hydro_gwh"),
                "nuclear": twh("nuclear_gwh"),
                "wind": twh("wind_gwh"),
                "solar": twh("solar_gwh"),
                "other": twh("other_gwh"),
                "D": twh("residual_demand_gwh"),
                "N": twh("net_import_gwh"),
                "inflow": twh("inflow_gwh"),
                "neg_inflow_wk": int((g["inflow_gwh"] < 0).sum()),
                "S_min": g["storage_gwh"].min(),
                "S_max": g["storage_gwh"].max(),
                "p_SE": mean_price("price_se_eur_mwh"),
                "p_SE1": mean_price("price_se1_eur_mwh"),
                "p_SE4": mean_price("price_se4_eur_mwh"),
                "p_DE": mean_price("price_de_eur_mwh"),
            }
        )
    table = pd.DataFrame(rows).set_index("year")
    lines.append(table.round(1).to_string())

    lines.append("\nPlausibility (warnings only; never used to tune anything)")
    for column, band in (("load", sanity.load_twh_per_year), ("hydro", sanity.hydro_twh_per_year)):
        outside = table[(table[column] < band[0]) | (table[column] > band[1])]
        status = "OK" if outside.empty else f"outside in {list(outside.index)}"
        lines.append(f"  {column} within {band} TWh/year: {status}")
    r_max = (
        cfg.reservoir.hydro_capacity_mw * HOURS_PER_WEEK * cfg.reservoir.availability / MWH_PER_GWH
    )
    hydro_stage = per_stage(w, ["hydro_gwh"])["hydro_gwh"]
    lines.append(
        f"  max weekly hydro {hydro_stage.max():,.0f} GWh vs R_max {r_max:,.0f} GWh "
        f"(weeks above: {int((hydro_stage > r_max).sum())})"
    )
    lines.append(
        f"  max storage {w['storage_gwh'].max():,.0f} GWh vs reference capacity "
        f"{sanity.storage_capacity_ref_gwh:,.0f} GWh"
    )
    pumping_share = w["pumping_gwh"].sum() / w["hydro_gwh"].sum()
    flag = "OK" if pumping_share <= sanity.max_pumping_share else "ABOVE LIMIT"
    lines.append(f"  pumping / hydro production: {pumping_share:.4f} ({flag})")
    ratio = w.groupby("iso_year").apply(
        lambda g: g["zone_load_gwh"].sum() / g["load_gwh"].sum(), include_groups=False
    )
    worst = float((ratio - 1).abs().max())
    flag = "OK" if worst <= sanity.zone_load_rel_tol else "ABOVE TOLERANCE"
    lines.append(f"  sum of zone loads vs SE load, worst year: {worst:.3%} ({flag})")
    inside = (
        w["price_se_eur_mwh"]
        >= w[[_zone_column(z) for z in cfg.data.price_zones]].min(axis=1) - 1e-6
    ) & (
        w["price_se_eur_mwh"]
        <= w[[_zone_column(z) for z in cfg.data.price_zones]].max(axis=1) + 1e-6
    )
    lines.append(
        f"  SE price between the zone prices: {int(inside.sum())} of "
        f"{int(w['price_se_eur_mwh'].notna().sum())} weeks"
    )

    lines.append("\nMissing values (NaN) per column")
    labels = [week_label(y, k) for y, k in zip(w["iso_year"], w["iso_week"], strict=True)]
    for column in WEEKLY_COLUMNS[4:]:
        missing = [lab for lab, m in zip(labels, w[column].isna(), strict=True) if m]
        if missing:
            lines.append(f"  {column}: {missing}")
    if interpolated:
        weeks = [d.strftime("%Y-%m-%d") for d in interpolated]
        lines.append(f"  storage interpolated at week boundaries (UTC Monday): {weeks}")

    lines.append(f"\nReservoir convention: configured {cfg.data.reservoir.label_semantics}")
    lines.append(references.to_string(index=False) if not references.empty else "  no references")
    roughness = {k: round(v, 1) for k, v in evidence.roughness_gwh.items()}
    lines.append(
        f"  winter roughness [GWh]: {roughness}, best {evidence.best.value}, "
        f"ratio {evidence.ratio:.2f}"
    )
    lines.append(f"  negative winter inflow weeks: {evidence.negative_weeks}")
    manifest = Manifest.load(raw_dir / "filelib" / "_manifest.json")
    lines.append(f"\nProvenance: {len(manifest.entries)} exports, digest {manifest.digest()[:16]}")
    return lines


def load_known_gaps(path: Path) -> list[dict]:
    """Read the documented gaps (``data/known_gaps.yaml``)."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or []


def unexplained_missing(weekly: pd.DataFrame, gaps: list[dict]) -> list[str]:
    """NaN cells not covered by a documented gap, as ``column@YYYY-Www``."""
    explained = {
        (column, week) for gap in gaps for column in gap["columns"] for week in gap["weeks"]
    }
    labels = [week_label(y, k) for y, k in zip(weekly["iso_year"], weekly["iso_week"], strict=True)]
    found = []
    for column in WEEKLY_COLUMNS[4:]:
        for label, missing in zip(labels, weekly[column].isna(), strict=True):
            if missing and (column, label) not in explained:
                found.append(f"{column}@{label}")
    return found


def write_weekly_csv(weekly: pd.DataFrame, path: Path, float_format: str) -> None:
    """Write the dataset with a fixed column order, UTF-8 and LF line endings."""
    out = weekly.copy()
    out["week_start"] = pd.DatetimeIndex(out["week_start"]).strftime("%Y-%m-%d")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, float_format=float_format, lineterminator="\n", encoding="utf-8")


def load_weekly(path: Path) -> pd.DataFrame:
    """Read ``weekly_se.csv`` and check the column contract.

    Returns
    -------
    pandas.DataFrame
        ``week_start`` as UTC timestamps, energies [GWh] and prices [EUR/MWh].
    """
    weekly = pd.read_csv(path, encoding="utf-8")
    if tuple(weekly.columns) != WEEKLY_COLUMNS:
        raise ValueError(f"{path} does not follow the weekly column contract")
    weekly["week_start"] = pd.to_datetime(weekly["week_start"], utc=True)
    return weekly
