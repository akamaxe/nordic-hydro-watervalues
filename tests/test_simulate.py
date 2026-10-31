"""Tests for the forward simulation, the backtest and the scenarios (offline)."""

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from watervalues.cli import app
from watervalues.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from watervalues.data.build import load_weekly
from watervalues.sdp import SdpProblem, build_problem, solve
from watervalues.simulate import (
    SimulationError,
    backtest_metrics,
    backtest_year,
    de_price_scenarios,
    dry_and_wet_years,
    load_metrics,
    simulate,
    water_value_on_path,
)
from watervalues.units import MODEL_WEEKS_PER_YEAR

CFG = load_config()
# A coarse storage grid keeps the backtest tests fast; the mechanics do not depend on it.
COARSE = CFG.model_copy(update={"sdp": CFG.sdp.model_copy(update={"n_storage_points": 51})})
STEP = CFG.simulation.decision_step_gwh


@pytest.fixture(scope="module")
def weekly() -> pd.DataFrame:
    return load_weekly(PROJECT_ROOT / CFG.data.weekly_csv)


@pytest.fixture(scope="module")
def base(weekly):
    return solve(build_problem(COARSE, weekly), CFG.sdp.tol_eur_mwh, CFG.sdp.max_cycles)


def run(solution, inflow=None, start=None, step=STEP) -> pd.DataFrame:
    p = solution.problem
    return simulate(
        solution,
        p.inflow_gwh.mean(axis=1) if inflow is None else inflow,
        p.residual_demand_gwh,
        p.intercept_eur_mwh,
        p.slope_eur_mwh_per_gwh,
        p.capacity_gwh / 2 if start is None else start,
        step,
    )


# --- Definition of Done (guide section 8, M5) ---------------------------------------------------


def test_mass_balance_closes_exactly_every_week(base) -> None:
    for inflow in (base.problem.inflow_gwh[:, 0], base.problem.inflow_gwh[:, -1]):
        trace = run(base, inflow)
        balance = trace["storage_gwh"] + trace["inflow_gwh"] - trace["release_gwh"]
        np.testing.assert_allclose(
            trace["storage_end_gwh"], balance - trace["spill_gwh"], atol=1e-9
        )
        np.testing.assert_array_equal(
            trace["storage_gwh"].iloc[1:], trace["storage_end_gwh"].iloc[:-1]
        )


# --- Simulation mechanics -----------------------------------------------------------------------


def test_simulation_respects_the_bounds(base) -> None:
    p = base.problem
    trace = run(base, p.inflow_gwh.max(axis=1), start=p.capacity_gwh)  # wettest weeks from full
    available = trace["storage_gwh"] + trace["inflow_gwh"]
    lower = np.minimum(p.release_min_gwh, available)
    assert (trace["release_gwh"] >= lower - 1e-9).all()
    assert (trace["release_gwh"] <= p.release_max_gwh + 1e-9).all()
    assert (trace["storage_end_gwh"] <= p.capacity_gwh + 1e-9).all()
    assert (trace["storage_end_gwh"] >= -1e-9).all()
    spilled = trace["spill_gwh"] > 0
    assert spilled.any()  # a full reservoir in the flood must spill
    np.testing.assert_allclose(trace.loc[spilled, "storage_end_gwh"], p.capacity_gwh)
    np.testing.assert_allclose(
        trace["price_eur_mwh"],
        p.intercept_eur_mwh + p.slope_eur_mwh_per_gwh * trace["net_import_gwh"],
    )


def test_simulation_follows_the_sdp_policy(base) -> None:
    # From a grid point, with the inflow of one scenario, the first decision matches the SDP's
    # decision up to one storage grid step (the SDP restricts the next storage to the grid).
    p = base.problem
    grid_step = p.storage_gwh[1]
    for k in (10, 25, 40):
        for j in range(p.inflow_gwh.shape[1]):
            trace = run(base, p.inflow_gwh[:, j], start=p.storage_gwh[k], step=1.0)
            assert abs(trace["release_gwh"].iloc[0] - base.release_gwh[0, k, j]) <= grid_step


def test_simulation_rejects_bad_inputs(base) -> None:
    with pytest.raises(SimulationError, match="finite"):
        run(base, np.full(MODEL_WEEKS_PER_YEAR - 1, 500.0))
    with pytest.raises(SimulationError, match="outside"):
        run(base, start=base.problem.capacity_gwh + 1)


def test_water_value_on_path_interpolates_each_week(base) -> None:
    path = np.linspace(10.0, 90.0, MODEL_WEEKS_PER_YEAR)
    w = water_value_on_path(base, path)
    s = base.problem.storage_gwh
    week = 20
    assert w[week - 1] == pytest.approx(
        np.interp(path[week - 1] / 100 * s[-1], s, base.water_value_eur_mwh[week - 1])
    )


def test_backtest_metrics_on_a_hand_made_trace() -> None:
    trace = pd.DataFrame(
        {
            "iso_year": [2019] * 4 + [2020] * 4,
            "price_eur_mwh": [10.0, 20.0, 30.0, 40.0, 50.0, 50.0, 50.0, 50.0],
            "observed_price_eur_mwh": [12.0, 18.0, 33.0, 37.0, 40.0, 60.0, np.nan, 99.0],
            "filled": [False] * 7 + [True],
            "storage_end_gwh": [100.0] * 8,
            "observed_storage_end_gwh": [100.0, 100.0, 100.0, 100.0, 90.0, 110.0, 90.0, 110.0],
            "spill_gwh": [0.0] * 7 + [5.0],
        }
    )
    m = backtest_metrics(trace, capacity_gwh=1_000.0)
    assert m.loc[2019, "price_mae_eur_mwh"] == pytest.approx(2.5)  # |-2|, |2|, |-3|, |3|
    assert m.loc[2019, "price_bias_eur_mwh"] == pytest.approx(0.0)
    assert m.loc[2019, "storage_rmse_pct"] == pytest.approx(0.0)
    assert m.loc[2020, "price_mae_eur_mwh"] == pytest.approx(
        10.0
    )  # the gap and the filled week out
    assert m.loc[2020, "storage_rmse_pct"] == pytest.approx(1.0)  # 10 GWh of 1000
    assert m.loc[2020, "spill_gwh"] == pytest.approx(5.0)
    assert m.loc["all", "price_mae_eur_mwh"] == pytest.approx((10 + 20) / 6)


# --- Backtest and scenarios on the committed dataset (coarse grid) -----------------------------


def test_backtest_year_never_sees_the_year(weekly) -> None:
    problem = build_problem(COARSE, weekly, inflow_years=[2016, 2017])
    assert problem.inflow_gwh.shape == (MODEL_WEEKS_PER_YEAR, 2)
    trace = backtest_year(COARSE, weekly, 2015)
    assert len(trace) == MODEL_WEEKS_PER_YEAR
    first = weekly[(weekly["iso_year"] == 2015) & (weekly["iso_week"] == 1)]
    assert trace["storage_gwh"].iloc[0] == pytest.approx(first["storage_gwh"].item())
    # 2015-W01 and 2015-W50 lack residual demand (and W01 the German price): filled, not scored
    filled = trace.loc[trace["filled"], "iso_week"].tolist()
    assert filled == [1, 50]


def test_higher_continental_prices_raise_water_values(weekly) -> None:
    table, slope = de_price_scenarios(COARSE, weekly)
    assert list(table.index) == [float(x) for x in CFG.scenarios.de_price_levels_eur_mwh]
    assert table["water_value_mean_eur_mwh"].is_monotonic_increasing
    assert slope > 0


def test_dry_year_ends_lower_than_wet_year(weekly, base) -> None:
    traces = dry_and_wet_years(COARSE, weekly, base)
    dry, wet = (traces[traces["case"] == case] for case in ("dry", "wet"))
    assert dry["storage_gwh"].iloc[0] == wet["storage_gwh"].iloc[0]  # same starting level
    assert dry["storage_end_gwh"].iloc[-1] < wet["storage_end_gwh"].iloc[-1]
    assert dry["price_eur_mwh"].mean() > wet["price_eur_mwh"].mean()


def _config_with(tmp_path, **sections: dict) -> str:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    for section, values in sections.items():
        raw[section].update(values)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(path)


def test_cli_simulate_writes_the_outputs(tmp_path) -> None:
    outputs = {
        name: str(tmp_path / f"{name}.csv")
        for name in ("backtest_csv", "metrics_csv", "de_price_csv", "dry_wet_csv")
    }
    outputs["report"] = str(tmp_path / "simulate_report.txt")
    config = _config_with(tmp_path, simulation=outputs, sdp={"n_storage_points": 21})
    result = CliRunner().invoke(app, ["simulate", "--config", config])
    assert result.exit_code == 0, result.output
    metrics = load_metrics(tmp_path / "metrics_csv.csv")
    years = list(range(CFG.period.first_iso_year, CFG.period.last_iso_year + 1))
    assert [int(y) for y in metrics.index[:-1]] == years
    assert metrics.index[-1] == "all"
    backtest = pd.read_csv(outputs["backtest_csv"])
    assert len(backtest) == len(years) * MODEL_WEEKS_PER_YEAR
    assert "slope of the mean water value" in (tmp_path / "simulate_report.txt").read_text(
        encoding="utf-8"
    )


def test_problem_inflow_years_must_exist(weekly) -> None:
    with pytest.raises(Exception, match="2040"):
        build_problem(COARSE, weekly, inflow_years=[2040])
    assert isinstance(build_problem(COARSE, weekly), SdpProblem)
