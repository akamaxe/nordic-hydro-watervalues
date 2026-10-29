"""Fetch raw data from the ENTSO-E File Library into the normalised cache."""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from watervalues.config import PROJECT_ROOT, AppConfig, DataConfig
from watervalues.data import exports
from watervalues.data.aggregate import RevisionStats, drop_revisions
from watervalues.data.cache import write_dataset
from watervalues.data.filelib import (
    FileClient,
    RateLimiter,
    call_with_retries,
    download_archives,
    make_client,
    select_files,
)

SOURCE = "filelib"
_DEC = 12


@dataclass(frozen=True)
class DatasetSpec:
    """How to read and deduplicate one dataset."""

    columns: tuple[str, ...]
    keys: tuple[str, ...]
    values: tuple[str, ...]


SPECS = {
    "generation": DatasetSpec(
        exports.GENERATION_COLUMNS,
        ("start_utc", "resolution_min", "psr_type"),
        ("value_mw", "consumption_mw"),
    ),
    "load": DatasetSpec(exports.LOAD_COLUMNS, ("start_utc", "resolution_min"), ("value_mw",)),
    "price": DatasetSpec(
        exports.PRICE_COLUMNS, ("start_utc", "resolution_min", "sequence"), ("price_eur_mwh",)
    ),
    "reservoir": DatasetSpec(exports.RESERVOIR_COLUMNS, ("iso_year", "iso_week"), ("stored_mwh",)),
}


@dataclass(frozen=True)
class AreaResult:
    """Outcome of fetching one dataset for one area."""

    rows: int
    revisions: RevisionStats


def dataset_areas(data: DataConfig, dataset: str) -> list[str]:
    """Areas kept for a dataset."""
    sweden = [data.country, *data.price_zones]
    if dataset == "price":
        return [*data.price_zones, data.de_price.before, data.de_price.after]
    if dataset == "reservoir":
        return sorted({data.reservoir.area, *sweden})
    return sweden


def period_bounds(cfg: AppConfig, dataset: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """First and last (year, month) of the exports needed for a dataset.

    The first ISO week of a year can start in December of the previous year. Reservoir values
    are also needed for the week before the first one (start-of-week storage) and for the year
    of the external reference points.
    """
    first, last = cfg.period.first_iso_year, cfg.period.last_iso_year
    if dataset == "reservoir":
        reference_years = [ref.iso_year for ref in cfg.data.reservoir.references]
        return (first - 1, 1), (max([last + 1, *reference_years]), _DEC)
    return (first - 1, _DEC), (last, _DEC)


def _parse(dataset: str, rows: pd.DataFrame, data: DataConfig) -> pd.DataFrame:
    if dataset == "generation":
        return exports.parse_generation(rows)
    if dataset == "load":
        return exports.parse_load(rows)
    if dataset == "price":
        return exports.parse_prices(rows, data.prices)
    return exports.parse_reservoir(rows)


def fetch_filelib(
    cfg: AppConfig,
    datasets: Sequence[str],
    *,
    force: bool = False,
    client: FileClient | None = None,
    progress: Callable[[str], None] = print,
) -> dict[str, dict[str, AreaResult]]:
    """Download the File Library exports and rebuild the normalised raw cache.

    Parameters
    ----------
    cfg : AppConfig
        Model configuration (period, areas, File Library settings).
    datasets : sequence of str
        Any of ``generation``, ``load``, ``price`` and ``reservoir``.
    force : bool
        Download again even if an export is cached.
    client : FileClient or None
        File Library client; ``None`` logs in with the stored website credentials.
    progress : callable
        Receives progress messages.

    Returns
    -------
    dict
        Per dataset and area, the number of rows cached and the revisions removed.
    """
    data = cfg.data
    settings = data.filelib
    raw_dir = PROJECT_ROOT / data.raw_dir
    unknown = sorted(set(datasets) - set(SPECS))
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}")
    client = client or make_client(settings.keyring_service, settings.timeout_s)
    limiter = RateLimiter(60 / settings.max_requests_per_minute)
    summary: dict[str, dict[str, AreaResult]] = {}
    for dataset in datasets:
        spec = SPECS[dataset]
        folder = settings.folders[dataset]
        listing: dict[str, Any] = call_with_retries(
            functools.partial(client.list_folder, folder),
            limiter,
            settings.max_retries,
            settings.backoff_s,
        )
        first, last = period_bounds(cfg, dataset)
        names = select_files(listing, first, last)
        progress(f"{dataset}: {len(names)} exports from {first} to {last}")
        paths = download_archives(
            client,
            folder,
            names,
            raw_dir,
            limiter,
            settings.max_retries,
            settings.backoff_s,
            force=force,
            progress=progress,
        )
        areas = dataset_areas(data, dataset)
        eics = [data.areas[a].eic for a in areas]
        parts: dict[str, list[pd.DataFrame]] = {a: [] for a in areas}
        for path in paths:
            raw = exports.read_export(path, spec.columns, eics, settings.read_chunk_rows)
            for area in areas:
                rows = exports.select_area(raw, data.areas[area])
                if len(rows):
                    parts[area].append(_parse(dataset, rows, data))
        summary[dataset] = {}
        for area in areas:
            if not parts[area]:
                progress(f"warning: no {dataset} data for {area}")
                continue
            frame, stats = drop_revisions(pd.concat(parts[area]), spec.keys, spec.values)
            years = frame["iso_year"] if dataset == "reservoir" else frame["start_utc"].dt.year
            write_dataset(frame, raw_dir, SOURCE, dataset, area, years)
            summary[dataset][area] = AreaResult(rows=len(frame), revisions=stats)
            progress(
                f"{dataset}/{area}: {len(frame):,} rows cached "
                f"({stats.duplicates} revisions dropped, {stats.conflicts} with changed values)"
            )
    return summary
