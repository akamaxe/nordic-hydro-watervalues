"""Load and validate the model configuration (``config/base.yaml``)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "base.yaml"


class _Section(BaseModel):
    """Base class for config sections: unknown keys are errors and values are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PeriodConfig(_Section):
    """Historical period used for calibration and scenarios."""

    start: date
    end: date

    @model_validator(mode="after")
    def _check_order(self) -> PeriodConfig:
        if self.end <= self.start:
            raise ValueError("period.end must be after period.start")
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


class AppConfig(_Section):
    """Full model configuration."""

    area: str
    period: PeriodConfig
    reservoir: ReservoirConfig
    price_model: PriceModelConfig
    sdp: SdpConfig
    simulation: SimulationConfig
    scenarios: ScenariosConfig


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
