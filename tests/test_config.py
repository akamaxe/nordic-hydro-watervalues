"""Tests for configuration validation."""

import copy

import pytest
import yaml
from pydantic import ValidationError

from watervalues.config import DEFAULT_CONFIG_PATH, AppConfig


@pytest.fixture
def raw() -> dict:
    with DEFAULT_CONFIG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_base_config_is_valid(raw: dict) -> None:
    cfg = AppConfig.model_validate(raw)
    assert cfg.data.de_price.switch_utc.utcoffset().total_seconds() == 0


def test_unknown_keys_are_rejected(raw: dict) -> None:
    bad = copy.deepcopy(raw)
    bad["sdp"]["n_storage_point"] = 101  # typo
    with pytest.raises(ValidationError, match="n_storage_point"):
        AppConfig.model_validate(bad)


def test_areas_must_be_defined(raw: dict) -> None:
    bad = copy.deepcopy(raw)
    del bad["data"]["areas"]["SE4"]
    with pytest.raises(ValidationError, match="SE4"):
        AppConfig.model_validate(bad)


def test_generation_categories_must_be_disjoint_known_codes(raw: dict) -> None:
    overlapping = copy.deepcopy(raw)
    overlapping["data"]["generation_categories"]["wind"] = ["B18", "B19", "B14"]
    with pytest.raises(ValidationError, match="more than one"):
        AppConfig.model_validate(overlapping)
    unknown = copy.deepcopy(raw)
    unknown["data"]["generation_categories"]["solar"] = ["B99"]
    with pytest.raises(ValidationError, match="B99"):
        AppConfig.model_validate(unknown)


def test_switch_time_needs_a_time_zone(raw: dict) -> None:
    bad = copy.deepcopy(raw)
    bad["data"]["de_price"]["switch_utc"] = "2018-09-30T22:00:00"
    with pytest.raises(ValidationError, match="time zone"):
        AppConfig.model_validate(bad)


def test_iso_period_order(raw: dict) -> None:
    bad = copy.deepcopy(raw)
    bad["period"] = {"first_iso_year": 2025, "last_iso_year": 2015}
    with pytest.raises(ValidationError, match="last_iso_year"):
        AppConfig.model_validate(bad)


def test_inflow_noise_band_must_be_positive(raw: dict) -> None:
    bad = copy.deepcopy(raw)
    bad["inflow"]["negative_noise_gwh"] = -1
    with pytest.raises(ValidationError, match="negative_noise_gwh"):
        AppConfig.model_validate(bad)


@pytest.mark.parametrize(
    "seasons",
    [
        {"Q1": 2, "Q2": 14},  # the first season must start in week 1
        {"Q1": 1, "Q2": 27, "Q3": 14},  # first weeks must increase
        {"Q1": 1, "Q2": 53},  # beyond the 52 model weeks
        {"Q1": 1},  # one season leaves nothing to compare
    ],
)
def test_price_seasons_start_at_week_one_and_increase(raw: dict, seasons: dict) -> None:
    bad = copy.deepcopy(raw)
    bad["price_model"]["seasons"] = seasons
    with pytest.raises(ValidationError, match="seasons"):
        AppConfig.model_validate(bad)


def test_price_base_sample_is_all_years_or_without_the_crisis(raw: dict) -> None:
    bad = copy.deepcopy(raw)
    bad["price_model"]["exclude_years"] = [2022]
    with pytest.raises(ValidationError, match="exclude_years"):
        AppConfig.model_validate(bad)


def test_price_slope_form_is_known(raw: dict) -> None:
    bad = copy.deepcopy(raw)
    bad["price_model"]["slope_form"] = "quadratic"
    with pytest.raises(ValidationError, match="slope_form"):
        AppConfig.model_validate(bad)
