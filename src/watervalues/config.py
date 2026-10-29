"""Load and validate the model configuration (``config/base.yaml``)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from watervalues.data.schema import KNOWN_PSR_TYPES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "base.yaml"
DATASETS = ("generation", "load", "price", "reservoir")


class _Section(BaseModel):
    """Base class for config sections: unknown keys are errors and values are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PeriodConfig(_Section):
    """Historical period used for calibration and scenarios, in ISO years."""

    first_iso_year: int
    last_iso_year: int

    @model_validator(mode="after")
    def _check_order(self) -> PeriodConfig:
        if self.last_iso_year < self.first_iso_year:
            raise ValueError("period.last_iso_year must not be before first_iso_year")
        return self


class ReservoirConfig(_Section):
    """Aggregated reservoir and hydro fleet parameters."""

    capacity_gwh: float | None = Field(default=None, gt=0)  # None -> max observed storage
    hydro_capacity_mw: float = Field(gt=0)
    availability: float = Field(gt=0, le=1)
    rmin_percentile: float = Field(ge=0, le=100)


class PriceModelConfig(_Section):
    """Options of the reduced-form price curve regression."""

    seasonal_dummies: bool
    exclude_years: tuple[int, ...] = ()


class SdpConfig(_Section):
    """Stochastic dynamic programming solver settings."""

    n_storage_points: int = Field(ge=3)
    discount_rate_annual: float = Field(ge=0)
    spill_penalty_eur_mwh: float = Field(ge=0)
    tol_eur_mwh: float = Field(gt=0)
    max_cycles: int = Field(ge=1)


class SimulationConfig(_Section):
    """Forward simulation settings."""

    decision_step_gwh: float = Field(gt=0)


class ScenariosConfig(_Section):
    """Scenario definitions."""

    de_price_levels_eur_mwh: tuple[float, ...]


class AreaSpec(_Section):
    """An ENTSO-E area: EIC code and the AreaTypeCode used in the bulk exports."""

    eic: str
    area_type: str | None = None  # None until checked against the exports


class DePriceConfig(_Section):
    """German price series across the 2018 bidding-zone split."""

    before: str
    after: str
    switch_utc: datetime

    @model_validator(mode="after")
    def _check_timezone(self) -> DePriceConfig:
        if self.switch_utc.tzinfo is None:
            raise ValueError("data.de_price.switch_utc must include a time zone")
        return self


class PriceFilterConfig(_Section):
    """Which rows of the price exports are the day-ahead auction price."""

    contract_type: str
    currency: str
    sequences: tuple[str, ...]


class ReferenceConfig(_Section):
    """External end-of-week storage observation [GWh]."""

    area: str
    iso_year: int
    iso_week: int
    stored_gwh: float


class ReservoirDataConfig(_Section):
    """Reservoir series and its labelling convention."""

    area: str
    label_semantics: Literal["start_of_week", "end_of_week", "weekly_mean"]
    max_interp_weeks: int = Field(ge=0)
    winter_weeks: tuple[int, int]
    max_shift_roughness_ratio: float = Field(gt=0, lt=1)  # configured vs one-week shift
    reference_tolerance_gwh: float = Field(gt=0)
    references: tuple[ReferenceConfig, ...] = ()


class SanityConfig(_Section):
    """Plausibility bands reported by the build (warnings only, never used for tuning)."""

    load_twh_per_year: tuple[float, float]
    hydro_twh_per_year: tuple[float, float]
    storage_capacity_ref_gwh: float = Field(gt=0)
    zone_load_rel_tol: float = Field(gt=0)
    max_pumping_share: float = Field(ge=0)


class ApiConfig(_Section):
    """ENTSO-E REST API access through entsoe-py."""

    env_file: Path
    retry_count: int = Field(ge=1)  # attempts per request (entsoe-py makes none with 0)
    retry_delay_s: float = Field(ge=0)
    timeout_s: float = Field(gt=0)
    max_retries: int = Field(ge=0)
    backoff_s: float = Field(ge=0)


class FileLibConfig(_Section):
    """ENTSO-E File Library access (website login, no API token)."""

    keyring_service: str
    token_url: str
    client_id: str
    timeout_s: float = Field(gt=0)
    max_requests_per_minute: int = Field(gt=0, le=100)
    max_retries: int = Field(ge=0)
    backoff_s: float = Field(ge=0)
    read_chunk_rows: int = Field(gt=0)
    folders: dict[str, str]

    @model_validator(mode="after")
    def _check_folders(self) -> FileLibConfig:
        if set(self.folders) != set(DATASETS):
            raise ValueError(f"data.filelib.folders must define exactly {DATASETS}")
        return self


class EnergiforetagenConfig(_Section):
    """Energiföretagen weekly statistics, used to validate the reservoir data."""

    dir: Path
    report: Path
    max_storage_gwh: float = Field(gt=0)
    timeout_s: float = Field(gt=0)
    workbooks: dict[int, str]


class DataConfig(_Section):
    """Data sources, areas and weekly aggregation rules."""

    source: Literal["filelib", "entsoe_api"]
    raw_dir: Path
    weekly_csv: Path
    build_report: Path
    known_gaps_file: Path
    min_coverage: float = Field(gt=0, le=1)
    csv_float_format: str
    country: str
    price_zones: tuple[str, ...]
    areas: dict[str, AreaSpec]
    de_price: DePriceConfig
    prices: PriceFilterConfig
    generation_categories: dict[str, tuple[str, ...]]
    partial_period_types: tuple[str, ...] = ()  # zero outside their reporting span
    reservoir: ReservoirDataConfig
    sanity: SanityConfig
    api: ApiConfig
    filelib: FileLibConfig
    energiforetagen: EnergiforetagenConfig

    @model_validator(mode="after")
    def _check_references(self) -> DataConfig:
        named = {
            self.country,
            *self.price_zones,
            self.de_price.before,
            self.de_price.after,
            self.reservoir.area,
            *(ref.area for ref in self.reservoir.references),
        }
        missing = sorted(named - set(self.areas))
        if missing:
            raise ValueError(f"areas used but not defined in data.areas: {missing}")
        codes = [code for codes in self.generation_categories.values() for code in codes]
        unknown = sorted((set(codes) | set(self.partial_period_types)) - set(KNOWN_PSR_TYPES))
        if unknown:
            raise ValueError(f"unknown production types in generation_categories: {unknown}")
        if len(codes) != len(set(codes)):
            raise ValueError("a production type appears in more than one generation category")
        return self


class AppConfig(_Section):
    """Full model configuration."""

    area: str
    period: PeriodConfig
    reservoir: ReservoirConfig
    price_model: PriceModelConfig
    sdp: SdpConfig
    simulation: SimulationConfig
    scenarios: ScenariosConfig
    data: DataConfig


def load_config(path: Path | None = None) -> AppConfig:
    """Read and validate a YAML configuration file.

    Parameters
    ----------
    path : Path or None
        YAML file to read. ``None`` reads ``config/base.yaml`` in the project root.

    Returns
    -------
    AppConfig
        Validated, immutable configuration.
    """
    config_path = DEFAULT_CONFIG_PATH if path is None else Path(path)
    with config_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return AppConfig.model_validate(raw)
