"""End-to-end test of the weekly build on a synthetic raw cache (ISO years 2019-2020)."""

import numpy as np
import pandas as pd
import pytest

from watervalues.config import load_config
from watervalues.data.build import (
    build_weekly,
    load_weekly,
    unexplained_missing,
    write_weekly_csv,
)
from watervalues.data.cache import write_dataset
from watervalues.data.schema import WEEKLY_COLUMNS

ZONE_SHARE = {"SE1": 0.1, "SE2": 0.2, "SE3": 0.6, "SE4": 0.1}
ZONE_PRICE = {"SE1": 20.0, "SE2": 22.0, "SE3": 40.0, "SE4": 50.0}
LOAD_MW = 15_000.0


def hourly_frame(start, end, value, **extra) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    return pd.DataFrame(
        {
            "start_utc": idx,
            "resolution_min": 60,
            **extra,
            "value": value if np.ndim(value) else np.full(len(idx), value),
        }
    )


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    rng = np.random.default_rng(1)
    start, end = pd.Timestamp("2018-12-31", tz="UTC"), pd.Timestamp("2021-01-04", tz="UTC")
    weeks = pd.date_range(start, end, freq="7D", inclusive="left")  # ISO 2019-W01 .. 2020-W53
    n = len(weeks)
    iso_week = np.asarray(weeks.isocalendar()["week"])
    hydro_gwh = 1500 + rng.normal(0, 200, n)
    inflow_gwh = 400 + 2500 * np.exp(-(((iso_week - 24) / 4) ** 2)) + rng.normal(0, 10, n)
    storage = 20_000 + np.concatenate([[0.0], np.cumsum(inflow_gwh - hydro_gwh)])

    hours = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    week_of_hour = (hours - start) // pd.Timedelta(days=7)
    hydro_mw = hydro_gwh[week_of_hour] * 1000 / 168

    def gen(psr, mw):
        return pd.DataFrame(
            {
                "start_utc": hours,
                "resolution_min": 60,
                "psr_type": psr,
                "value_mw": mw if np.ndim(mw) else np.full(len(hours), mw),
                "consumption_mw": 0.0,
            }
        )

    generation = pd.concat(
        [gen("B12", hydro_mw), gen("B14", 6000.0), gen("B19", 3000.0), gen("B20", 1000.0)],
        ignore_index=True,
    )
    write_dataset(generation, raw, "filelib", "generation", "SE", generation["start_utc"].dt.year)

    def load(mw):
        return pd.DataFrame({"start_utc": hours, "resolution_min": 60, "value_mw": mw})

    write_dataset(load(LOAD_MW), raw, "filelib", "load", "SE", hours.year)
    for zone, share in ZONE_SHARE.items():
        write_dataset(load(LOAD_MW * share), raw, "filelib", "load", zone, hours.year)
        prices = pd.DataFrame(
            {"start_utc": hours, "resolution_min": 60, "price_eur_mwh": ZONE_PRICE[zone]}
        )
        write_dataset(prices, raw, "filelib", "price", zone, hours.year)
    de = pd.DataFrame({"start_utc": hours, "resolution_min": 60, "price_eur_mwh": 60.0})
    write_dataset(de, raw, "filelib", "price", "DE_LU", hours.year)
    old_de = de.iloc[:1].assign(start_utc=pd.Timestamp("2018-01-01", tz="UTC"))
    write_dataset(old_de, raw, "filelib", "price", "DE_AT_LU", [2018])

    # End-of-week labels: 2018-W52 (start of 2019-W01) to 2020-W53.
    label_weeks = pd.date_range("2018-12-24", periods=n + 1, freq="7D", tz="UTC")
    iso = label_weeks.isocalendar()
    labels = pd.DataFrame(
        {
            "iso_year": np.asarray(iso["year"]),
            "iso_week": np.asarray(iso["week"]),
            "stored_mwh": storage * 1000.0,
        }
    )
    write_dataset(labels, raw, "filelib", "reservoir", "SE", labels["iso_year"])

    cfg = load_config()
    reservoir = cfg.data.reservoir.model_copy(update={"references": ()})
    data = cfg.data.model_copy(update={"raw_dir": raw, "reservoir": reservoir})
    period = cfg.period.model_copy(update={"first_iso_year": 2019, "last_iso_year": 2020})
    cfg = cfg.model_copy(update={"data": data, "period": period})
    return cfg, inflow_gwh, hydro_gwh


def test_build_follows_the_contract(synthetic) -> None:
    cfg, _, _ = synthetic
    weekly = build_weekly(cfg).weekly
    assert tuple(weekly.columns) == WEEKLY_COLUMNS
    assert len(weekly) == 104
    assert weekly["hours"].tolist().count(336) == 1  # 2020-W52 absorbs W53
    assert weekly.loc[weekly["hours"] == 336, "iso_year"].item() == 2020
    assert not weekly.drop(columns="solar_gwh").isna().any().any()
    assert (weekly["solar_gwh"] == 0).all()  # no solar type in this area: structural zero


def test_storage_is_continuous_and_inflow_is_recovered(synthetic) -> None:
    cfg, inflow, hydro = synthetic
    weekly = build_weekly(cfg).weekly
    np.testing.assert_allclose(weekly["storage_end_gwh"].iloc[:-1], weekly["storage_gwh"].iloc[1:])
    implied = weekly["storage_end_gwh"] - weekly["storage_gwh"] + weekly["hydro_gwh"]
    expected = np.append(inflow[:103], inflow[103] + inflow[104])  # folded 2020-W52 + W53
    np.testing.assert_allclose(implied, expected, atol=1e-6)
    np.testing.assert_allclose(weekly["hydro_gwh"].iloc[:103], hydro[:103])


def test_prices_and_energies(synthetic) -> None:
    cfg, _, _ = synthetic
    weekly = build_weekly(cfg).weekly
    expected_se = sum(ZONE_SHARE[z] * ZONE_PRICE[z] for z in ZONE_SHARE)
    np.testing.assert_allclose(weekly["price_se_eur_mwh"], expected_se)
    np.testing.assert_allclose(weekly["price_se3_eur_mwh"], 40.0)
    np.testing.assert_allclose(weekly["price_de_eur_mwh"], 60.0)
    np.testing.assert_allclose(weekly["load_gwh"] / weekly["hours"], LOAD_MW / 1000)
    np.testing.assert_allclose(weekly["nuclear_gwh"] / weekly["hours"], 6.0)


def test_reservoir_evidence_is_reported(synthetic) -> None:
    cfg, _, _ = synthetic
    result = build_weekly(cfg)
    assert result.evidence.best.value == "end_of_week"
    assert any("Reservoir convention" in line for line in result.report)


def test_csv_round_trip_and_missing_value_check(synthetic, tmp_path) -> None:
    cfg, _, _ = synthetic
    weekly = build_weekly(cfg).weekly
    path = tmp_path / "weekly.csv"
    write_weekly_csv(weekly, path, "%.3f")
    assert b"\r\n" not in path.read_bytes()
    loaded = load_weekly(path)
    np.testing.assert_allclose(loaded["storage_gwh"], weekly["storage_gwh"], atol=5e-4)
    assert loaded["week_start"].iloc[0] == pd.Timestamp("2018-12-31", tz="UTC")

    gappy = weekly.copy()
    gappy.loc[0, "hydro_gwh"] = np.nan
    assert unexplained_missing(gappy, []) == ["hydro_gwh@2019-W01"]
    documented = [{"weeks": ["2019-W01"], "columns": ["hydro_gwh"], "reason": "test"}]
    assert unexplained_missing(gappy, documented) == []
