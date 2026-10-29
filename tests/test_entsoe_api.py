"""Tests for the entsoe-py source (fake client shaped like entsoe-py 0.8.1 output)."""

import os

import dotenv
import numpy as np
import pandas as pd
import pytest
from entsoe.exceptions import NoMatchingDataError

from watervalues.config import load_config
from watervalues.data import entsoe_api
from watervalues.data.cache import read_dataset
from watervalues.data.entsoe_api import (
    ApiError,
    check_api,
    fetch_api,
    get_api_key,
    infer_resolution_min,
    normalise_generation,
    normalise_reservoir,
    normalise_series,
)
from watervalues.data.filelib import CredentialsError

TOKEN = "secret-token-123"


def test_resolution_follows_the_spacing_of_the_index() -> None:
    hourly = pd.date_range("2025-09-30 22:00", periods=2, freq="h", tz="UTC")
    quarter = pd.date_range("2025-10-01 00:00", periods=4, freq="15min", tz="UTC")
    gap_then_point = pd.DatetimeIndex(["2025-10-01 03:00", "2025-10-01 03:15"], tz="UTC")
    index = hourly.append(quarter).append(gap_then_point)
    assert infer_resolution_min(index).tolist() == [60, 60, 15, 15, 15, 15, 15, 15]


def test_generation_multiindex_and_flat_columns_normalise_alike() -> None:
    idx = pd.date_range("2025-01-01 00:00", periods=3, freq="h", tz="Europe/Stockholm")
    multi = pd.DataFrame(
        {
            ("Hydro Water Reservoir", "Actual Aggregated"): [8000.0, 8100.0, np.nan],
            ("Hydro Pumped Storage", "Actual Aggregated"): [1.0, 2.0, 3.0],
            ("Hydro Pumped Storage", "Actual Consumption"): [5.0, 0.0, 0.0],
            ("Hydro Run-of-river and poundage", "Actual Aggregated"): [10.0, 10.0, 10.0],
        },
        index=idx,
    )
    out = normalise_generation(multi)
    assert sorted(out["psr_type"].unique()) == ["B10", "B11", "B12"]  # poundage == pondage
    b10 = out[out["psr_type"] == "B10"]
    assert b10["consumption_mw"].tolist() == [5.0, 0.0, 0.0]
    assert out["start_utc"].iloc[0] == pd.Timestamp("2024-12-31 23:00", tz="UTC")
    assert np.isnan(out[out["psr_type"] == "B12"]["value_mw"].iloc[2])  # missing stays NaN
    flat = pd.DataFrame({"Nuclear": [6000.0, 6000.0, 6000.0]}, index=idx)
    nuclear = normalise_generation(flat)
    assert nuclear["psr_type"].unique().tolist() == ["B14"]
    assert (nuclear["consumption_mw"] == 0).all()
    assert (nuclear["resolution_min"] == 60).all()


def test_series_normalisation_converts_to_utc() -> None:
    idx = pd.date_range("2025-06-01 00:00", periods=2, freq="h", tz="Europe/Stockholm")
    out = normalise_series(pd.DataFrame({"Actual Load": [1.0, 2.0]}, index=idx), "value_mw")
    assert out["start_utc"].iloc[0] == pd.Timestamp("2025-05-31 22:00", tz="UTC")
    assert out["value_mw"].tolist() == [1.0, 2.0]


def test_reservoir_labels_survive_dst_drift() -> None:
    index = pd.DatetimeIndex(
        [
            "2026-10-11 22:00",  # Monday 2026-10-12 00:00 CEST -> 2026-W42
            "2026-10-18 22:00",  # Monday 00:00 CEST -> W43
            "2026-10-25 22:00",  # 7 days later in UTC = Sunday 23:00 CET (drift) -> still W44
        ],
        tz="UTC",
    )
    out = normalise_reservoir(pd.Series([1.0, 2.0, 3.0], index=index))
    assert list(zip(out["iso_year"], out["iso_week"], strict=True)) == [
        (2026, 42),
        (2026, 43),
        (2026, 44),
    ]


class FakeApi:
    """Deterministic client: one constant value per dataset, honouring the requested window."""

    def __init__(self, fail_with: Exception | None = None) -> None:
        self.calls: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
        self.fail_with = fail_with

    def _index(self, start, end, freq):
        if self.fail_with is not None:
            raise self.fail_with
        if end <= pd.Timestamp("2015-01-01", tz="UTC"):
            raise NoMatchingDataError  # no data before the platform started
        # entsoe-py returns local time and includes the end instant
        return pd.date_range(start, end, freq=freq, tz="UTC").tz_convert("Europe/Stockholm")

    def query_generation(self, country_code, *, start, end, **kwargs):
        self.calls.append(("generation", start, end))
        return pd.DataFrame({"Nuclear": 6000.0}, index=self._index(start, end, "h"))

    def query_load(self, country_code, *, start, end):
        self.calls.append(("load", start, end))
        return pd.DataFrame({"Actual Load": 15000.0}, index=self._index(start, end, "h"))

    def query_day_ahead_prices(self, country_code, *, start, end):
        self.calls.append(("price", start, end))
        return pd.Series(40.0, index=self._index(start, end, "h"))

    def query_aggregate_water_reservoirs_and_hydro_storage(self, country_code, *, start, end):
        self.calls.append(("reservoir", start, end))
        mondays = pd.date_range(start.ceil("D"), end, freq="W-MON", tz="UTC")
        local_midnight = mondays.tz_convert(None).tz_localize("Europe/Stockholm").tz_convert("UTC")
        values = 20_000_000.0 + np.arange(len(local_midnight))
        return pd.Series(values, index=local_midnight)


@pytest.fixture
def cfg(tmp_path):
    base = load_config()
    period = base.period.model_copy(update={"first_iso_year": 2015, "last_iso_year": 2016})
    reservoir = base.data.reservoir.model_copy(update={"references": ()})
    data = base.data.model_copy(update={"raw_dir": tmp_path, "reservoir": reservoir})
    return base.model_copy(update={"period": period, "data": data})


def test_fetch_api_writes_a_continuous_cache(cfg, tmp_path) -> None:
    client = FakeApi()
    fetch_api(cfg, ["load", "price", "reservoir"], client=client, progress=lambda _: None)
    load = read_dataset(tmp_path, "entsoe_api", "load", "SE")
    assert load["start_utc"].min() == pd.Timestamp("2015-01-01", tz="UTC")
    assert not load["start_utc"].duplicated().any()  # inclusive window ends were deduplicated
    assert load["start_utc"].max() == pd.Timestamp("2016-12-31 23:00", tz="UTC")
    prices = read_dataset(tmp_path, "entsoe_api", "price", "SE1")
    assert (prices["sequence"] == "").all()
    reservoir = read_dataset(tmp_path, "entsoe_api", "reservoir", "SE")
    labels = list(zip(reservoir["iso_year"], reservoir["iso_week"], strict=True))
    assert len(labels) == len(set(labels))  # overlapping windows deduplicated
    weeks = pd.DatetimeIndex([pd.Timestamp.fromisocalendar(y, w, 1) for y, w in sorted(labels)])
    steps = np.diff(weeks.as_unit("ns").asi8)
    assert (steps == pd.Timedelta(days=7).value).all()  # no missing week
    first_window = next(c for c in client.calls if c[0] == "load")
    assert first_window[1] == pd.Timestamp("2014-12-29", tz="UTC")  # start of ISO 2015-W01


def test_generation_is_queried_for_the_country_only(cfg) -> None:
    client = FakeApi()
    summary = fetch_api(cfg, ["generation"], client=client, progress=lambda _: None)
    assert list(summary["generation"]) == ["SE"]


def test_errors_are_redacted(cfg) -> None:
    leaky = RuntimeError(f"401 Client Error for url: https://x/api?securityToken={TOKEN}")
    with pytest.raises(ApiError) as excinfo:
        fetch_api(cfg, ["load"], client=FakeApi(leaky), api_key=TOKEN, progress=lambda _: None)
    assert TOKEN not in str(excinfo.value)
    assert "***" in str(excinfo.value)
    with pytest.raises(ApiError) as excinfo:
        check_api(cfg, client=FakeApi(leaky), api_key=TOKEN)
    assert TOKEN not in str(excinfo.value)


def test_check_api_counts_prices(cfg) -> None:
    assert check_api(cfg, client=FakeApi()) == 25  # 24 hours plus the inclusive end


def test_api_key_comes_from_env_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(os, "environ", os.environ.copy())  # keep the test's env changes local
    monkeypatch.setattr(entsoe_api, "load_dotenv", dotenv.load_dotenv)
    env_file = tmp_path / ".env"
    env_file.write_text(f"ENTSOE_API_KEY={TOKEN}\n", encoding="utf-8")
    assert get_api_key(env_file) == TOKEN


def test_missing_api_key_explains_what_to_do(tmp_path) -> None:
    with pytest.raises(CredentialsError, match="ENTSOE_API_KEY"):
        get_api_key(tmp_path / ".env")
