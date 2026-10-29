"""Shared data definitions: production-type codes, area codes and the weekly dataset contract."""

from __future__ import annotations

from typing import Final

from entsoe.mappings import PSRTYPE_MAPPINGS

# ENTSO-E production types (PSR codes B01-B25) and their published names.
PSR_NAMES: Final[dict[str, str]] = {
    code: name for code, name in PSRTYPE_MAPPINGS.items() if code.startswith("B")
}
KNOWN_PSR_TYPES: Final[tuple[str, ...]] = tuple(sorted(PSR_NAMES))

# Spelling variants seen in ENTSO-E files and entsoe-py versions.
_NAME_ALIASES: Final[dict[str, str]] = {"poundage": "pondage"}


def _normalise_name(name: str) -> str:
    text = " ".join(name.strip().lower().split())
    for old, new in _NAME_ALIASES.items():
        text = text.replace(old, new)
    return text


_CODE_BY_NAME: Final[dict[str, str]] = {_normalise_name(n): c for c, n in PSR_NAMES.items()}


def psr_code(name: str) -> str:
    """Return the PSR code of a production-type name (e.g. ``"Nuclear"`` -> ``"B14"``).

    Raises
    ------
    KeyError
        If the name is not an ENTSO-E production type.
    """
    try:
        return _CODE_BY_NAME[_normalise_name(name)]
    except KeyError:
        raise KeyError(f"unknown production type name: {name!r}") from None


# Column contract of data/processed/weekly_se.csv (guide section 4, plus `hours` and
# `storage_end_gwh`; see data/README.md).
WEEKLY_COLUMNS: Final[tuple[str, ...]] = (
    "week_start",
    "iso_year",
    "iso_week",
    "hours",
    "storage_gwh",
    "storage_end_gwh",
    "hydro_gwh",
    "load_gwh",
    "nuclear_gwh",
    "wind_gwh",
    "solar_gwh",
    "other_gwh",
    "price_se_eur_mwh",
    "price_se1_eur_mwh",
    "price_se2_eur_mwh",
    "price_se3_eur_mwh",
    "price_se4_eur_mwh",
    "price_de_eur_mwh",
)
ENERGY_COLUMNS: Final[tuple[str, ...]] = tuple(
    c for c in WEEKLY_COLUMNS if c.endswith("_gwh") and not c.startswith("storage")
)
