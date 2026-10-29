"""Parse ENTSO-E File Library exports into normalised frames.

Exports are tab-separated UTF-8 files inside zips, one per month (one per year for reservoirs),
covering every area. Each parser keeps one area and returns a normalised frame:

* interval datasets (generation, load, price): ``start_utc`` [UTC], ``resolution_min`` [min], the
  value column(s) and ``update_utc`` (publication time, used to drop revisions);
* reservoir: ``iso_year``, ``iso_week`` (the published label), ``stored_mwh`` [MWh] and
  ``update_utc``.
"""

from __future__ import annotations

import re
import zipfile
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from watervalues.config import AreaSpec, PriceFilterConfig
from watervalues.data.schema import psr_code

_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
_RESOLUTION = re.compile(r"^PT(?P<minutes>\d+)M$")

GENERATION_COLUMNS = (
    "DateTime(UTC)",
    "ResolutionCode",
    "AreaCode",
    "AreaTypeCode",
    "ProductionType",
    "ActualGenerationOutput[MW]",
    "ActualConsumption[MW]",
    "UpdateTime(UTC)",
)
LOAD_COLUMNS = (
    "DateTime(UTC)",
    "ResolutionCode",
    "AreaCode",
    "AreaTypeCode",
    "TotalLoad[MW]",
    "UpdateTime(UTC)",
)
PRICE_COLUMNS = (
    "DateTime(UTC)",
    "ResolutionCode",
    "AreaCode",
    "AreaTypeCode",
    "ContractType",
    "Sequence",
    "Price[Currency/MWh]",
    "Currency",
    "UpdateTime(UTC)",
)
RESERVOIR_COLUMNS = (
    "Year",
    "Week",
    "AreaCode",
    "AreaTypeCode",
    "StoredEnergy[MWh]",
    "UpdateTime(UTC)",
)


def read_export(
    path: Path, columns: Sequence[str], area_codes: Sequence[str], chunk_rows: int
) -> pd.DataFrame:
    """Read the export inside a File Library zip, keeping only some areas.

    The file is read in chunks and filtered as it goes, so memory stays bounded even for the
    generation exports (over a million rows per month, all areas).

    Parameters
    ----------
    path : Path
        Downloaded zip.
    columns : sequence of str
        Columns to read; all are read as text.
    area_codes : sequence of str
        EIC codes to keep.
    chunk_rows : int
        Rows read at a time.

    Returns
    -------
    pandas.DataFrame
        Text columns of the matching rows.
    """
    with zipfile.ZipFile(path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"expected one CSV in {path.name}, found {members}")
        with zf.open(members[0]) as fh:
            reader = pd.read_csv(
                fh,
                sep="\t",
                encoding="utf-8-sig",
                usecols=list(columns),
                dtype=str,
                keep_default_na=False,
                chunksize=chunk_rows,
            )
            kept = [chunk[chunk["AreaCode"].isin(area_codes)] for chunk in reader]
    if not kept:
        return pd.DataFrame(columns=list(columns))
    return pd.concat(kept, ignore_index=True)


def select_area(df: pd.DataFrame, spec: AreaSpec) -> pd.DataFrame:
    """Rows of one area, filtered by EIC code and AreaTypeCode.

    With ``spec.area_type = None`` the area must appear with a single AreaTypeCode; otherwise
    rows of different area types (for example a country also published as a control area) could
    be counted twice.
    """
    rows = df[df["AreaCode"] == spec.eic]
    if spec.area_type is not None:
        return rows[rows["AreaTypeCode"] == spec.area_type]
    types = sorted(rows["AreaTypeCode"].unique())
    if len(types) > 1:
        raise ValueError(f"area {spec.eic} appears with several AreaTypeCodes {types}; set one")
    return rows


def resolution_minutes(codes: pd.Series) -> np.ndarray:
    """Convert ISO 8601 resolution codes such as ``PT15M`` to minutes."""
    extracted = codes.str.extract(_RESOLUTION)["minutes"]
    if extracted.isna().any():
        bad = sorted(codes[extracted.isna()].unique())
        raise ValueError(f"unsupported resolution codes: {bad}")
    return extracted.astype(int).to_numpy()


def _times(text: pd.Series) -> pd.arrays.DatetimeArray:
    """Parse UTC timestamps, keeping a timezone-aware array (no object conversion)."""
    return pd.to_datetime(text, format=_TIME_FORMAT, utc=True).array


def _numbers(text: pd.Series) -> np.ndarray:
    """Parse numbers; markers such as ``n/e`` or empty cells become NaN."""
    return pd.to_numeric(text, errors="coerce").to_numpy(dtype=float)


def parse_generation(rows: pd.DataFrame) -> pd.DataFrame:
    """Normalise generation rows of one area (production types as PSR codes, power in MW)."""
    names = rows["ProductionType"]
    codes = {name: psr_code(name) for name in names.unique()}
    return pd.DataFrame(
        {
            "start_utc": _times(rows["DateTime(UTC)"]),
            "resolution_min": resolution_minutes(rows["ResolutionCode"]),
            "psr_type": names.map(codes).to_numpy(),
            "value_mw": _numbers(rows["ActualGenerationOutput[MW]"]),
            "consumption_mw": _numbers(rows["ActualConsumption[MW]"]),
            "update_utc": _times(rows["UpdateTime(UTC)"]),
        }
    )


def parse_load(rows: pd.DataFrame) -> pd.DataFrame:
    """Normalise total-load rows of one area (power in MW)."""
    return pd.DataFrame(
        {
            "start_utc": _times(rows["DateTime(UTC)"]),
            "resolution_min": resolution_minutes(rows["ResolutionCode"]),
            "value_mw": _numbers(rows["TotalLoad[MW]"]),
            "update_utc": _times(rows["UpdateTime(UTC)"]),
        }
    )


def parse_prices(rows: pd.DataFrame, price_filter: PriceFilterConfig) -> pd.DataFrame:
    """Normalise day-ahead price rows of one area [EUR/MWh].

    Only the configured contract type, currency and auction sequences are kept. Text fields are
    compared without surrounding spaces: rows without an auction sequence carry a single space.

    Raises
    ------
    ValueError
        If a kept row is in another currency, so that no price is silently mixed.
    """
    sequence = rows["Sequence"].str.strip()
    contract = rows["ContractType"].str.strip()
    keep = (contract == price_filter.contract_type) & sequence.isin(price_filter.sequences)
    kept = rows[keep]
    currency = kept["Currency"].str.strip()
    other_currency = currency != price_filter.currency
    if other_currency.any():
        found = sorted(currency[other_currency].unique())
        raise ValueError(f"day-ahead prices in unexpected currencies: {found}")
    return pd.DataFrame(
        {
            "start_utc": _times(kept["DateTime(UTC)"]),
            "resolution_min": resolution_minutes(kept["ResolutionCode"]),
            "sequence": sequence[keep].to_numpy(),
            "price_eur_mwh": _numbers(kept["Price[Currency/MWh]"]),
            "update_utc": _times(kept["UpdateTime(UTC)"]),
        }
    )


def parse_reservoir(rows: pd.DataFrame) -> pd.DataFrame:
    """Normalise reservoir rows of one area (ISO week labels, stored energy in MWh).

    Raises
    ------
    ValueError
        If a label is not a valid ISO week.
    """
    years = rows["Year"].astype(int).to_numpy()
    weeks = rows["Week"].astype(int).to_numpy()
    for year, week in zip(years, weeks, strict=True):
        date.fromisocalendar(int(year), int(week), 1)  # raises for an invalid ISO week
    return pd.DataFrame(
        {
            "iso_year": years,
            "iso_week": weeks,
            "stored_mwh": _numbers(rows["StoredEnergy[MWh]"]),
            "update_utc": _times(rows["UpdateTime(UTC)"]),
        }
    )
