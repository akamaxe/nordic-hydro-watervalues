"""Tests for the inflow reconstruction and the scenario matrix (synthetic data, offline)."""

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from watervalues.cli import app
from watervalues.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, PeriodConfig, load_config
from watervalues.data.build import load_known_gaps, load_weekly, week_label, write_weekly_csv
from watervalues.data.calendar import week_calendar
from watervalues.data.schema import WEEKLY_COLUMNS
from watervalues.inflow import (
    InflowError,
    annual_totals,
    build_inflow,
    climatology,
    dry_and_wet,
    fill_gaps,
    load_scenarios,
    reconstruct,
    scenario_matrix,
)
from watervalues.units import HOURS_PER_WEEK, MODEL_WEEKS_PER_YEAR

CFG = load_config()
NOISE_GWH = CFG.inflow.negative_noise_gwh
FIRST, LAST = 2019, 2021  # 2020 has an ISO week 53, folded into week 52 (336 h)
SYNTHETIC_CFG = CFG.model_copy(
    update={"period": PeriodConfig(first_iso_year=FIRST, last_iso_year=LAST)}
)
STORAGE0_GWH = 20_000.0


def synthetic_weekly(inflow_gwh: np.ndarray, hydro_gwh: np.ndarray) -> pd.DataFrame:
    """Weekly dataset whose storage follows the energy balance of the given flows exactly.

    ``inflow_gwh`` and ``hydro_gwh`` are totals over each row's hours (336 h for the folded
    week); integer values keep every sum exact in floating point.
    """
    cal = week_calendar(FIRST, LAST)
    storage = STORAGE0_GWH + np.concatenate([[0.0], np.cumsum(inflow_gwh - hydro_gwh)])
    columns = {name: np.full(len(cal), 1.0) for name in WEEKLY_COLUMNS}
    columns.update(
        week_start=cal["week_start"].to_numpy(),
        iso_year=cal["iso_year"].to_numpy(),
        iso_week=cal["iso_week"].to_numpy(),
        hours=cal["hours"].to_numpy(),
        storage_gwh=storage[:-1],
        storage_end_gwh=storage[1:],
        hydro_gwh=hydro_gwh,
    )
    return pd.DataFrame(columns)[list(WEEKLY_COLUMNS)]


def week_mask(cal: pd.DataFrame, iso_year: int, iso_week: int) -> np.ndarray:
    return ((cal["iso_year"] == iso_year) & (cal["iso_week"] == iso_week)).to_numpy()


@pytest.fixture
def flows() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    cal = week_calendar(FIRST, LAST)
    rng = np.random.default_rng(1)
    inflow = rng.integers(300, 4_000, len(cal)).astype(float)
    hydro = rng.integers(800, 1_900, len(cal)).astype(float)
    return cal, inflow, hydro


def test_synthetic_inflow_is_recovered_exactly(flows) -> None:
    cal, inflow, hydro = flows
    out = reconstruct(synthetic_weekly(inflow, hydro), NOISE_GWH)
    np.testing.assert_array_equal(out["balance_gwh"], inflow)
    np.testing.assert_array_equal(out["inflow_stage_gwh"], inflow * HOURS_PER_WEEK / cal["hours"])
    assert (out["status"] == "observed").all()
    # The folded week: the balance spans 336 hours and is halved only afterwards.
    folded = (cal["hours"] == 2 * HOURS_PER_WEEK).to_numpy()
    assert folded.sum() == 1
    np.testing.assert_array_equal(out.loc[folded, "inflow_stage_gwh"], inflow[folded] / 2)


def test_small_negatives_are_clipped_and_listed(flows) -> None:
    _, inflow, hydro = flows
    inflow[10] = -40.0  # 2019-W11
    inflow[20] = -NOISE_GWH  # 2019-W21: the edge of the band is still noise
    out = reconstruct(synthetic_weekly(inflow, hydro), NOISE_GWH)
    assert out.loc[10, "balance_gwh"] == -40.0  # the measured balance is kept
    assert out.loc[[10, 20], "inflow_stage_gwh"].tolist() == [0.0, 0.0]
    assert out.index[out["status"] == "clipped"].tolist() == [10, 20]
    report = "\n".join(build_inflow(SYNTHETIC_CFG, synthetic_weekly(inflow, hydro)).report)
    assert "2019-W11" in report
    assert "2019-W21" in report


def test_negative_beyond_the_noise_band_stops(flows) -> None:
    _, inflow, hydro = flows
    inflow[10] = -(NOISE_GWH + 250.0)
    with pytest.raises(InflowError, match="2019-W11"):
        reconstruct(synthetic_weekly(inflow, hydro), NOISE_GWH)


def test_noise_band_applies_before_scaling(flows) -> None:
    cal, inflow, hydro = flows
    folded = (cal["hours"] == 2 * HOURS_PER_WEEK).to_numpy()
    inflow[folded] = -(NOISE_GWH + 50.0)  # beyond the band over 336 h, inside it per stage
    with pytest.raises(InflowError, match="2020-W52"):
        reconstruct(synthetic_weekly(inflow, hydro), NOISE_GWH)


def test_gap_week_gets_the_mean_of_the_clipped_stage_inflow(flows) -> None:
    cal, inflow, hydro = flows
    gap, folded, clipped = (week_mask(cal, year, 52) for year in (2019, 2020, 2021))
    inflow[clipped] = -40.0
    weekly = synthetic_weekly(inflow, hydro)
    weekly.loc[gap, "hydro_gwh"] = np.nan
    reconstructed = reconstruct(weekly, NOISE_GWH)
    assert reconstructed.loc[gap, "status"].item() == "missing"
    filled = fill_gaps(reconstructed)
    # The same week of the other years per 168-hour stage: half the folded 2020-W52, and
    # 2021-W52 clipped to zero. Averaging the raw balances would give a different value.
    expected = (inflow[folded].item() / 2 + 0.0) / 2
    assert filled.loc[gap, "inflow_stage_gwh"].item() == pytest.approx(expected)
    assert expected != pytest.approx((inflow[folded].item() + inflow[clipped].item()) / 2)
    assert filled.loc[gap, "status"].item() == "imputed"
    assert np.isnan(filled.loc[gap, "balance_gwh"].item())  # nothing measured is invented
    others = ~gap
    np.testing.assert_array_equal(
        filled.loc[others, "inflow_stage_gwh"], reconstructed.loc[others, "inflow_stage_gwh"]
    )
    assert (filled.loc[others, "status"] == reconstructed.loc[others, "status"]).all()


def test_only_weeks_without_production_are_imputed(flows) -> None:
    _, inflow, hydro = flows
    weekly = synthetic_weekly(inflow, hydro)
    weekly.loc[30, "storage_end_gwh"] = np.nan  # 2019-W31: production known, storage not
    with pytest.raises(InflowError, match="2019-W31"):
        fill_gaps(reconstruct(weekly, NOISE_GWH))


def test_scenario_matrix_is_years_by_stages(flows) -> None:
    _, inflow, hydro = flows
    out = fill_gaps(reconstruct(synthetic_weekly(inflow, hydro), NOISE_GWH))
    matrix = scenario_matrix(out)
    assert matrix.shape == (LAST - FIRST + 1, MODEL_WEEKS_PER_YEAR)
    assert matrix.index.tolist() == list(range(FIRST, LAST + 1))
    assert matrix.columns.tolist() == list(range(1, MODEL_WEEKS_PER_YEAR + 1))
    np.testing.assert_array_equal(matrix.to_numpy().ravel(), out["inflow_stage_gwh"].to_numpy())


def test_scenario_matrix_rejects_incomplete_years(flows) -> None:
    _, inflow, hydro = flows
    out = fill_gaps(reconstruct(synthetic_weekly(inflow, hydro), NOISE_GWH))
    with pytest.raises(InflowError, match="2020"):
        scenario_matrix(out.drop(index=60))  # 2020 loses a week
    missing = out.copy()
    missing.loc[5, "inflow_stage_gwh"] = np.nan
    with pytest.raises(InflowError, match="2019-W06"):
        scenario_matrix(missing)


def test_load_scenarios_rejects_a_broken_matrix(tmp_path, flows) -> None:
    _, inflow, hydro = flows
    matrix = scenario_matrix(fill_gaps(reconstruct(synthetic_weekly(inflow, hydro), NOISE_GWH)))
    path = tmp_path / "scenarios.csv"
    matrix.to_csv(path)
    pd.testing.assert_frame_equal(load_scenarios(path), matrix)
    matrix.drop(columns=[52]).to_csv(path)
    with pytest.raises(InflowError, match="stages"):
        load_scenarios(path)


def test_annual_totals_rank_years_by_52_standard_weeks() -> None:
    cal = week_calendar(FIRST, LAST)
    per_week = cal["iso_year"].map({2019: 1_010.0, 2020: 1_000.0, 2021: 900.0})
    inflow = (per_week * cal["hours"] / HOURS_PER_WEEK).to_numpy()
    hydro = (1_000.0 * cal["hours"] / HOURS_PER_WEEK).to_numpy()
    weekly = synthetic_weekly(inflow, hydro)
    gap = week_mask(cal, 2021, 10)
    weekly.loc[gap, "hydro_gwh"] = np.nan
    annual = annual_totals(fill_gaps(reconstruct(weekly, NOISE_GWH)))
    # The ISO year keeps all 53 weeks of 2020; the model year has 52 standard weeks.
    assert annual.loc[2020, "inflow_iso_twh"] == pytest.approx(53 * 1.0)
    assert annual.loc[2020, "inflow_model_twh"] == pytest.approx(52 * 1.0)
    assert annual.loc[2019, "inflow_model_twh"] == pytest.approx(52 * 1.01)
    assert annual.loc[2020, "hydro_iso_twh"] == pytest.approx(53 * 1.0)
    assert annual.loc[2020, "hydro_model_twh"] == pytest.approx(52 * 1.0)
    # 2021: the imputed week (mean of 1010 and 1000 GWh) counts in the inflow; the production
    # of the year is unknown.
    assert annual.loc[2021, "inflow_model_twh"] == pytest.approx((51 * 900 + 1_005) / 1_000)
    assert annual.loc[2021, "imputed_weeks"] == 1
    assert annual.loc[2021, ["hydro_iso_twh", "hydro_model_twh"]].isna().all()
    assert annual["wet_rank"].tolist() == [1, 2, 3]
    # 2020 has the largest ISO-year total but 2019 the wettest model year.
    assert dry_and_wet(annual) == (2021, 2019)


def test_climatology_spans_the_years(flows) -> None:
    _, inflow, hydro = flows
    matrix = scenario_matrix(fill_gaps(reconstruct(synthetic_weekly(inflow, hydro), NOISE_GWH)))
    clim = climatology(matrix)
    assert clim.index.tolist() == list(range(1, MODEL_WEEKS_PER_YEAR + 1))
    assert clim.columns.tolist() == ["min", "mean", "max"]
    assert (clim["min"] <= clim["mean"]).all()
    assert (clim["mean"] <= clim["max"]).all()
    np.testing.assert_allclose(clim["mean"], matrix.mean(axis=0))


def test_scenario_years_must_match_the_period(flows) -> None:
    _, inflow, hydro = flows
    with pytest.raises(InflowError, match="period"):
        build_inflow(CFG, synthetic_weekly(inflow, hydro))  # data 2019-2021, period 2015-2025


def test_committed_dataset_gives_a_complete_scenario_matrix() -> None:
    weekly = load_weekly(PROJECT_ROOT / CFG.data.weekly_csv)
    result = build_inflow(CFG, weekly)
    years = CFG.period.last_iso_year - CFG.period.first_iso_year + 1
    assert result.scenarios.shape == (years, MODEL_WEEKS_PER_YEAR)
    assert result.scenarios.notna().all().all()
    assert (result.scenarios >= 0).all().all()
    # Exactly the documented weeks without hydro production are imputed.
    gaps = load_known_gaps(PROJECT_ROOT / CFG.data.known_gaps_file)
    expected = sorted(w for gap in gaps if "hydro_gwh" in gap["columns"] for w in gap["weeks"])
    imputed = result.weekly[result.weekly["status"] == "imputed"]
    labels = sorted(
        week_label(y, k) for y, k in zip(imputed["iso_year"], imputed["iso_week"], strict=True)
    )
    assert labels == expected
    assert set(result.weekly["status"]) <= {"observed", "clipped", "imputed"}
    unknown_production = result.annual.index[result.annual["hydro_iso_twh"].isna()]
    assert unknown_production.tolist() == sorted(set(imputed["iso_year"]))


def _config_with(tmp_path, **sections: dict) -> str:
    """Write a copy of the base config with some section values replaced; return its path."""
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    for section, values in sections.items():
        raw[section].update(values)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(path)


def test_cli_inflow_writes_the_outputs(tmp_path) -> None:
    outputs = {
        "weekly_csv": str(tmp_path / "inflow_weekly.csv"),
        "scenarios_csv": str(tmp_path / "inflow_scenarios.csv"),
        "report": str(tmp_path / "inflow_report.txt"),
    }
    config = _config_with(tmp_path, inflow=outputs)
    result = CliRunner().invoke(app, ["inflow", "--config", config])
    assert result.exit_code == 0, result.output
    expected = build_inflow(CFG, load_weekly(PROJECT_ROOT / CFG.data.weekly_csv)).scenarios
    pd.testing.assert_frame_equal(
        load_scenarios(outputs["scenarios_csv"]), expected, check_exact=False, atol=1e-3
    )
    weekly = pd.read_csv(outputs["weekly_csv"])
    assert {"balance_gwh", "inflow_stage_gwh", "status"} <= set(weekly.columns)
    assert "imputed" in (tmp_path / "inflow_report.txt").read_text(encoding="utf-8")


def test_cli_inflow_stops_on_a_large_negative(tmp_path, flows) -> None:
    _, inflow, hydro = flows
    inflow[10] = -(NOISE_GWH + 250.0)
    csv = tmp_path / "weekly.csv"
    write_weekly_csv(synthetic_weekly(inflow, hydro), csv, CFG.data.csv_float_format)
    config = _config_with(
        tmp_path,
        period={"first_iso_year": FIRST, "last_iso_year": LAST},
        data={"weekly_csv": str(csv)},
        inflow={"report": str(tmp_path / "report.txt")},
    )
    result = CliRunner().invoke(app, ["inflow", "--config", config])
    assert result.exit_code == 1
    assert "2019-W11" in result.output
    assert not (tmp_path / "report.txt").exists()
