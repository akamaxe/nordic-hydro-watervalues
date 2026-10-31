"""End-to-end test of the weekly build on a synthetic raw cache (ISO years 2019-2020)."""

import warnings

import numpy as np
import pandas as pd
import pytest

from watervalues.config import ReferenceConfig, load_config
from watervalues.data.aggregate import hourly_energy, per_stage
from watervalues.data.build import (
    BuildError,
    build_weekly,
    data_digest,
    derived_columns,
    load_weekly,
    reported_spans,
    sanity_report,
    unexplained_missing,
    unfolded_hydro,
    write_weekly_csv,
)
from watervalues.data.cache import read_dataset, write_dataset
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


@pytest.fixture(scope="module")
def built(synthetic):
    """One build of the synthetic cache, shared by the tests that only read it."""
    cfg, _, _ = synthetic
    return build_weekly(cfg)


def test_build_follows_the_contract(built) -> None:
    weekly = built.weekly
    assert tuple(weekly.columns) == WEEKLY_COLUMNS
    assert len(weekly) == 104
    assert weekly["hours"].tolist().count(336) == 1  # 2020-W52 absorbs W53
    assert weekly.loc[weekly["hours"] == 336, "iso_year"].item() == 2020
    assert not weekly.drop(columns="solar_gwh").isna().any().any()
    assert (weekly["solar_gwh"] == 0).all()  # no solar type in this area: structural zero


def test_storage_is_continuous_and_inflow_is_recovered(synthetic, built) -> None:
    _, inflow, hydro = synthetic
    weekly = built.weekly
    np.testing.assert_allclose(weekly["storage_end_gwh"].iloc[:-1], weekly["storage_gwh"].iloc[1:])
    implied = weekly["storage_end_gwh"] - weekly["storage_gwh"] + weekly["hydro_gwh"]
    expected = np.append(inflow[:103], inflow[103] + inflow[104])  # folded 2020-W52 + W53
    np.testing.assert_allclose(implied, expected, atol=1e-6)
    np.testing.assert_allclose(weekly["hydro_gwh"].iloc[:103], hydro[:103])


def test_prices_and_energies(built) -> None:
    weekly = built.weekly
    expected_se = sum(ZONE_SHARE[z] * ZONE_PRICE[z] for z in ZONE_SHARE)
    np.testing.assert_allclose(weekly["price_se_eur_mwh"], expected_se)
    np.testing.assert_allclose(weekly["price_se3_eur_mwh"], 40.0)
    np.testing.assert_allclose(weekly["price_de_eur_mwh"], 60.0)
    np.testing.assert_allclose(weekly["load_gwh"] / weekly["hours"], LOAD_MW / 1000)
    np.testing.assert_allclose(weekly["nuclear_gwh"] / weekly["hours"], 6.0)


def test_reservoir_evidence_is_reported(built) -> None:
    result = built
    assert result.evidence.best.value == "end_of_week"
    assert any("Reservoir convention" in line for line in result.report)
    assert any("vs its one-week shift (start_of_week)" in line for line in result.report)
    assert any(line.startswith("\nProvenance: source filelib") for line in result.report)
    assert "  B12 Hydro Water Reservoir: 2018-12-31 00:00 to 2021-01-03 23:00" in result.report


def test_report_checks_prices_and_zone_loads_on_comparable_weeks(synthetic, built) -> None:
    cfg, _, _ = synthetic
    result = built
    assert any(
        line.endswith("between the zones' load-weighted prices: 104 of 104 weeks")
        for line in result.report
    )
    weekly = result.weekly.copy()
    zone_load = weekly["load_gwh"].to_numpy().copy()
    weekly.loc[0, "load_gwh"] = np.nan  # SE load missing in a week that the zones cover
    zone_price = np.tile([ZONE_PRICE[z] for z in ZONE_SHARE], (len(weekly), 1))
    zone_price[1] = np.nan  # a week without zone prices: no bound, and no warning
    weekly.loc[1, "price_se_eur_mwh"] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lines = sanity_report(
            cfg,
            weekly,
            np.zeros(len(weekly)),
            zone_load,
            zone_price,
            pd.DataFrame(columns=["min", "max"]),
            result.references,
            result.evidence,
            [],
            "test",
        )
    zone_line = next(line for line in lines if "sum of zone loads" in line)
    assert zone_line.endswith("0.000% (OK)")
    assert any(line.endswith("load-weighted prices: 103 of 103 weeks") for line in lines)


def test_unfolded_hydro_keeps_week_53_separate(synthetic) -> None:
    cfg, _, hydro = synthetic
    generation = read_dataset(cfg.data.raw_dir, "filelib", "generation", "SE")
    series = unfolded_hydro(cfg, hourly_energy(generation, "value_mw", by=("psr_type",)))
    assert len(series) == 52 + 53
    np.testing.assert_allclose(series.to_numpy(), hydro)


def test_data_digest_depends_on_content_not_on_order() -> None:
    a = pd.DataFrame({"x": [1.0, 2.0]})
    b = pd.DataFrame({"y": [3.0]})
    assert data_digest({("load", "SE"): a, ("price", "SE1"): b}) == data_digest(
        {("price", "SE1"): b, ("load", "SE"): a}
    )
    assert data_digest({("load", "SE"): a}) != data_digest({("load", "SE"): a + 1})


def test_csv_round_trip_and_missing_value_check(built, tmp_path) -> None:
    weekly = built.weekly
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


def with_references(cfg, *references: ReferenceConfig):
    reservoir = cfg.data.reservoir.model_copy(update={"references": references})
    return cfg.model_copy(update={"data": cfg.data.model_copy(update={"reservoir": reservoir})})


def test_reference_weeks_must_be_cached_and_must_match(synthetic) -> None:
    cfg, _, _ = synthetic
    labels = read_dataset(cfg.data.raw_dir, "filelib", "reservoir", "SE")
    week = labels[(labels["iso_year"] == 2019) & (labels["iso_week"] == 10)]
    stored_gwh = float(week["stored_mwh"].iloc[0]) / 1000
    build_weekly(
        with_references(
            cfg, ReferenceConfig(area="SE", iso_year=2019, iso_week=10, stored_gwh=stored_gwh)
        )
    )
    with pytest.raises(BuildError, match="contradict"):
        build_weekly(
            with_references(
                cfg,
                ReferenceConfig(area="SE", iso_year=2019, iso_week=10, stored_gwh=stored_gwh + 50),
            )
        )
    with pytest.raises(BuildError, match="not in the cached reservoir data"):
        build_weekly(
            with_references(
                cfg, ReferenceConfig(area="SE", iso_year=2030, iso_week=10, stored_gwh=stored_gwh)
            )
        )


def test_reported_spans_ignore_rows_without_a_value() -> None:
    times = pd.date_range("2021-01-01", periods=4, freq="h", tz="UTC")
    generation = pd.DataFrame(
        {"psr_type": "B16", "start_utc": times, "value_mw": [np.nan, 5.0, 6.0, np.nan]}
    )
    spans = reported_spans(generation)
    assert spans.loc["B16", "min"] == times[1]
    assert spans.loc["B16", "max"] == times[2]


def test_inflow_of_a_folded_week_is_computed_before_scaling(synthetic, built) -> None:
    _, inflow, _ = synthetic
    weekly = derived_columns(built.weekly)
    stage = per_stage(weekly, ["hydro_gwh", "inflow_gwh"])
    folded = weekly["hours"] == 336
    np.testing.assert_allclose(
        stage.loc[folded, "inflow_gwh"], (inflow[103] + inflow[104]) / 2, atol=1e-6
    )
    # Scaling first would leave a 336-hour storage change against 168 hours of production.
    wrong = derived_columns(per_stage(weekly, ["hydro_gwh"]))
    assert not np.allclose(wrong.loc[folded, "inflow_gwh"], stage.loc[folded, "inflow_gwh"])
