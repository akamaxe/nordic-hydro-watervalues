"""Tests for the forward simulation, the backtest and the scenarios (offline)."""

import dataclasses

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from watervalues import simulate as simulate_module
from watervalues.cli import app
from watervalues.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from watervalues.data.build import load_weekly
from watervalues.sdp import SdpError, SdpProblem, build_problem, solve
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


def run(solution, inflow=None, start=None, step=STEP, demand=None) -> pd.DataFrame:
    p = solution.problem
    return simulate(
        solution,
        p.inflow_gwh.mean(axis=1) if inflow is None else inflow,
        p.residual_demand_gwh if demand is None else demand,
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


def test_a_full_reservoir_spills_rather_than_sells_below_minus_the_spill_penalty(base) -> None:
    # With a full reservoir, a big inflow and a low demand the price curve would go negative
    # before full release: production stops near where the price reaches -c_sp.
    p = base.problem
    inflow = np.full(MODEL_WEEKS_PER_YEAR, 3_000.0)
    demand = np.full(MODEL_WEEKS_PER_YEAR, 200.0)
    trace = run(base, inflow, start=p.capacity_gwh, step=1.0, demand=demand)
    week = 0
    no_loss = (
        demand[week]
        + (p.intercept_eur_mwh[week] + p.spill_penalty_eur_mwh) / (p.slope_eur_mwh_per_gwh[week])
    )
    assert no_loss < p.release_max_gwh[week]
    assert trace["spill_gwh"].iloc[week] > 0
    assert trace["release_gwh"].iloc[week] == pytest.approx(no_loss, abs=1.0)  # 1 GWh grid
    assert (
        trace["price_eur_mwh"].iloc[week]
        >= -p.spill_penalty_eur_mwh - 1.0 * (p.slope_eur_mwh_per_gwh[week])
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
    negative = np.full(MODEL_WEEKS_PER_YEAR, 500.0)
    negative[3] = -10.0
    with pytest.raises(SimulationError, match=">= 0"):
        run(base, negative)


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
            "curve_price_eur_mwh": [11.0, 19.0, 33.0, 37.0, 45.0, 55.0, 50.0, 50.0],
            "filled": [False] * 7 + [True],
            "folded": [False] * 7 + [True],
            "storage_end_gwh": [100.0] * 8,
            "observed_storage_end_gwh": [100.0, 100.0, 100.0, 100.0, 90.0, 110.0, 90.0, 500.0],
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
    # The bias splits into the curve's error and the effect of operating differently.
    assert m.loc[2020, "curve_bias_eur_mwh"] == pytest.approx(((45 - 40) + (55 - 60)) / 2)
    assert m.loc[2020, "operating_bias_eur_mwh"] == pytest.approx(((50 - 45) + (50 - 55)) / 2)
    total = m.loc[2020, "curve_bias_eur_mwh"] + m.loc[2020, "operating_bias_eur_mwh"]
    assert total == pytest.approx(m.loc[2020, "price_bias_eur_mwh"])
    assert m.loc[2020, "storage_rmse_pct"] == pytest.approx(1.0)  # the folded week is left out
    assert m.loc[2020, "spill_gwh"] == pytest.approx(5.0)
    assert m.loc["all", "price_mae_eur_mwh"] == pytest.approx((10 + 20) / 6)
    assert np.isnan(m.loc[2019, "price_corr_annual_means"])  # only on the pooled row


# --- Backtest and scenarios on the committed dataset (coarse grid) -----------------------------


def test_backtest_year_never_sees_the_year(weekly, monkeypatch) -> None:
    calls = []
    real_build_problem = simulate_module.build_problem

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real_build_problem(*args, **kwargs)

    monkeypatch.setattr(simulate_module, "build_problem", spy)
    run_2015 = backtest_year(COARSE, weekly, 2015)
    year_problem = next(call for call in calls if call.get("curve") is not None)
    assert 2015 not in year_problem["inflow_years"]
    assert len(year_problem["inflow_years"]) == 10
    assert 2015 in year_problem["curve"].spec.exclude_years
    assert year_problem["residual_demand_gwh"] is not None  # the year's own demand
    trace = run_2015.trace
    assert len(trace) == MODEL_WEEKS_PER_YEAR
    first = weekly[(weekly["iso_year"] == 2015) & (weekly["iso_week"] == 1)]
    assert trace["storage_gwh"].iloc[0] == pytest.approx(first["storage_gwh"].item())
    # 2015-W01 and 2015-W50 lack residual demand (and W01 the German price): filled, not scored
    assert trace.loc[trace["filled"], "iso_week"].tolist() == [1, 50]
    assert trace.loc[trace["folded"], "iso_week"].tolist() == [52]  # 2015 has an ISO week 53
    assert run_2015.checks["converged"] == 1.0


def test_build_problem_reuses_the_table_and_inflow(weekly) -> None:
    from watervalues.inflow import build_inflow
    from watervalues.price import price_table

    table = price_table(weekly, CFG.price_model.seasons)
    inflow = build_inflow(CFG, weekly)
    fresh = build_problem(COARSE, weekly)
    reused = build_problem(COARSE, weekly, table=table, inflow=inflow)
    for field in dataclasses.fields(SdpProblem):
        np.testing.assert_array_equal(getattr(fresh, field.name), getattr(reused, field.name))
    demand = np.full(MODEL_WEEKS_PER_YEAR, 900.0)
    own = build_problem(COARSE, weekly, residual_demand_gwh=demand, table=table, inflow=inflow)
    np.testing.assert_array_equal(own.residual_demand_gwh, demand)


def test_higher_continental_prices_raise_water_values(weekly) -> None:
    table, slope, solutions = de_price_scenarios(COARSE, weekly)
    assert list(table.index) == [float(x) for x in CFG.scenarios.de_price_levels_eur_mwh]
    assert table["water_value_mean_eur_mwh"].is_monotonic_increasing
    assert slope > 0
    assert all(solution.converged for solution in solutions.values())


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
    names = ("backtest_csv", "metrics_csv", "de_price_csv", "dry_wet_csv", "checks_csv")
    outputs = {name: str(tmp_path / f"{name}.csv") for name in names}
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
    found = pd.read_csv(outputs["checks_csv"], index_col="solution")
    assert len(found) == len(years) + len(CFG.scenarios.de_price_levels_eur_mwh) + 1
    assert (found["converged"] == 1).all()
    report = (tmp_path / "simulate_report.txt").read_text(encoding="utf-8")
    assert "slope of the mean water value" in report
    assert "Checks of the water values" in report


def test_problem_inflow_years_must_exist(weekly) -> None:
    with pytest.raises(SdpError, match="2040"):
        build_problem(COARSE, weekly, inflow_years=[2040])
