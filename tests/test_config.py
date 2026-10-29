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
