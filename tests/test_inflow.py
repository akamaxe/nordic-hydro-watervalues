"""Tests for the inflow reconstruction and the scenario matrix (synthetic data, offline)."""

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from watervalues.cli import app
from watervalues.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
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
    reconstruct,
    scenario_matrix,
)
from watervalues.units import HOURS_PER_WEEK, MODEL_WEEKS_PER_YEAR

CFG = load_config()
NOISE_GWH = CFG.inflow.negative_noise_gwh
FIRST, LAST = 2019, 2021  # 2020 has an ISO week 53, folded into week 52 (336 h)
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
    np.testing.assert_array_equal(out["inflow_gwh"], inflow * HOURS_PER_WEEK / cal["hours"])
    assert (out["status"] == "observed").all()
    # The folded week: the balance spans 336 hours and is halved only afterwards.
    folded = (cal["hours"] == 2 * HOURS_PER_WEEK).to_numpy()
    assert folded.sum() == 1
    np.testing.assert_array_equal(out.loc[folded, "inflow_gwh"], inflow[folded] / 2)


def test_small_negative_is_clipped_and_listed(flows) -> None:
    _, inflow, hydro = flows
    inflow[10] = -40.0  # 2019-W11
    out = reconstruct(synthetic_weekly(inflow, hydro), NOISE_GWH)
    assert out.loc[10, "balance_gwh"] == -40.0  # the measured balance is kept
    assert out.loc[10, "inflow_gwh"] == 0.0
    assert out["status"].tolist().count("clipped") == 1
    assert out.loc[10, "status"] == "clipped"
    report = "\n".join(build_inflow(CFG, synthetic_weekly(inflow, hydro)).report)
    assert "2019-W11" in report


def test_negative_beyond_the_noise_band_stops(flows) -> None:
    _, inflow, hydro = flows
    inflow[10] = -(NOISE_GWH + 250.0)
    with pytest.raises(InflowError, match="2019-W11"):
        reconstruct(synthetic_weekly(inflow, hydro), NOISE_GWH)


def test_gap_week_gets_the_mean_of_the_same_iso_week(flows) -> None:
    cal, inflow, hydro = flows
    weekly = synthetic_weekly(inflow, hydro)
    gap = ((weekly["iso_year"] == 2020) & (weekly["iso_week"] == 10)).to_numpy()
    weekly.loc[gap, "hydro_gwh"] = np.nan
    complete = reconstruct(synthetic_weekly(inflow, hydro), NOISE_GWH)
    filled = fill_gaps(reconstruct(weekly, NOISE_GWH))
    same_week = (cal["iso_week"] == 10).to_numpy() & ~gap
    assert filled.loc[gap, "inflow_gwh"].item() == pytest.approx(inflow[same_week].mean())
    assert filled.loc[gap, "status"].item() == "imputed"
    assert np.isnan(filled.loc[gap, "balance_gwh"].item())  # nothing measured is invented
    np.testing.assert_array_equal(filled.loc[~gap, "inflow_gwh"], complete.loc[~gap, "inflow_gwh"])
    assert (filled.loc[~gap, "status"] == "observed").all()


def test_scenario_matrix_is_years_by_stages(flows) -> None:
    _, inflow, hydro = flows
    out = fill_gaps(reconstruct(synthetic_weekly(inflow, hydro), NOISE_GWH))
    matrix = scenario_matrix(out)
    assert matrix.shape == (LAST - FIRST + 1, MODEL_WEEKS_PER_YEAR)
    assert matrix.index.tolist() == list(range(FIRST, LAST + 1))
    assert matrix.columns.tolist() == list(range(1, MODEL_WEEKS_PER_YEAR + 1))
    np.testing.assert_array_equal(matrix.to_numpy().ravel(), out["inflow_gwh"].to_numpy())


def test_scenario_matrix_rejects_incomplete_years(flows) -> None:
    _, inflow, hydro = flows
    out = fill_gaps(reconstruct(synthetic_weekly(inflow, hydro), NOISE_GWH))
    with pytest.raises(InflowError, match="2020"):
        scenario_matrix(out.drop(index=60))  # 2020 loses a week
    missing = out.copy()
    missing.loc[5, "inflow_gwh"] = np.nan
    with pytest.raises(InflowError, match="2019-W06"):
        scenario_matrix(missing)


def test_annual_totals_rank_years_by_52_standard_weeks() -> None:
    cal = week_calendar(FIRST, LAST)
    per_week = cal["iso_year"].map({2019: 1_010.0, 2020: 1_000.0, 2021: 900.0})
    inflow = (per_week * cal["hours"] / HOURS_PER_WEEK).to_numpy()
    hydro = (1_000.0 * cal["hours"] / HOURS_PER_WEEK).to_numpy()
    annual = annual_totals(fill_gaps(reconstruct(synthetic_weekly(inflow, hydro), NOISE_GWH)))
    # The ISO year keeps all 53 weeks of 2020; the model year has 52 standard weeks.
    assert annual.loc[2020, "inflow_iso_twh"] == pytest.approx(53 * 1.0)
    assert annual.loc[2020, "inflow_model_twh"] == pytest.approx(52 * 1.0)
    assert annual.loc[2019, "inflow_model_twh"] == pytest.approx(52 * 1.01)
    assert annual.loc[2020, "hydro_iso_twh"] == pytest.approx(53 * 1.0)
    assert annual.loc[2020, "hydro_model_twh"] == pytest.approx(52 * 1.0)
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
    scenarios = pd.read_csv(outputs["scenarios_csv"], index_col="iso_year")
    years = CFG.period.last_iso_year - CFG.period.first_iso_year + 1
    assert scenarios.shape == (years, MODEL_WEEKS_PER_YEAR)
    weekly = pd.read_csv(outputs["weekly_csv"])
    assert {"balance_gwh", "inflow_gwh", "status"} <= set(weekly.columns)
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
