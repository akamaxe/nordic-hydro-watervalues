"""Tests for the water value SDP (synthetic problems and the committed dataset, offline)."""

import time

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from watervalues.cli import app
from watervalues.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from watervalues.data.build import load_weekly
from watervalues.price import de_price_profile, price_table, quadratic_cost_keur
from watervalues.sdp import (
    SdpError,
    SdpProblem,
    backward_pass,
    base_de_price_profile,
    build_problem,
    checks,
    compare_solutions,
    de_price_levels,
    first_order_check,
    grid_resolution_eur_mwh,
    grid_sensitivity,
    load_solution,
    release_min,
    solve,
    water_value_at,
    water_values,
)
from watervalues.units import HOURS_PER_WEEK, MODEL_WEEKS_PER_YEAR, MWH_PER_GWH

CFG = load_config()
SDP = CFG.sdp


def toy_problem(
    stages: int = 4,
    scenarios: int = 3,
    points: int = 21,
    capacity_gwh: float = 2_000.0,
    discount: float = 0.99,
    seed: int = 0,
) -> SdpProblem:
    """A small random problem: inflow, demand and price curve of plausible magnitudes."""
    rng = np.random.default_rng(seed)
    return SdpProblem(
        storage_gwh=np.linspace(0.0, capacity_gwh, points),
        inflow_gwh=rng.uniform(0.0, 600.0, (stages, scenarios)),
        probability=np.full(scenarios, 1.0 / scenarios),
        residual_demand_gwh=rng.uniform(800.0, 1_200.0, stages),
        release_min_gwh=rng.uniform(50.0, 150.0, stages),
        release_max_gwh=np.full(stages, 900.0),
        intercept_eur_mwh=rng.uniform(30.0, 60.0, stages),
        slope_eur_mwh_per_gwh=rng.uniform(0.03, 0.08, stages),
        spill_penalty_eur_mwh=0.01,
        discount=discount,
    )


def rough_problem(scenarios: int = 2, seed: int = 1) -> SdpProblem:
    """A small problem where the bounds bind: big inflows that force spill from a full
    reservoir, and a price curve that goes below zero well before full release."""
    rng = np.random.default_rng(seed)
    inflow = rng.uniform(100.0, 900.0, (4, scenarios))
    inflow[1] = rng.uniform(2_000.0, 2_600.0, scenarios)  # a flood week
    return SdpProblem(
        storage_gwh=np.linspace(0.0, 2_000.0, 21),
        inflow_gwh=inflow,
        probability=np.full(scenarios, 1.0 / scenarios),
        residual_demand_gwh=np.full(4, 800.0),
        release_min_gwh=np.full(4, 100.0),
        release_max_gwh=np.full(4, 1_500.0),
        intercept_eur_mwh=rng.uniform(10.0, 20.0, 4),
        slope_eur_mwh_per_gwh=np.full(4, 0.05),  # price < 0 beyond about 1,000 GWh of release
        spill_penalty_eur_mwh=0.01,
        discount=0.99,
    )


def loop_dp(problem: SdpProblem, terminal_keur: np.ndarray) -> np.ndarray:
    """Expected cost-to-go by plain loops, minimising over the release/spill split by brute force.

    For each storage now, scenario and storage next, the water let out ``X`` is fixed; the
    release is searched on a dense grid between its bounds and the rest is spilled.
    Independent of the solver's vectorisation and of its closed-form split.
    """
    s = problem.storage_gwh
    stages, scenarios = problem.inflow_gwh.shape
    value = np.zeros((stages + 1, len(s)))
    value[stages] = terminal_keur
    for t in reversed(range(stages)):
        for k, storage in enumerate(s):
            for j in range(scenarios):
                available = storage + problem.inflow_gwh[t, j]
                best = np.inf
                for k2, storage_next in enumerate(s):
                    let_out = available - storage_next
                    lower = min(problem.release_min_gwh[t], available)
                    upper = min(let_out, problem.release_max_gwh[t])
                    if let_out < lower or upper < lower:
                        continue
                    release = np.linspace(lower, upper, 2_001)
                    cost = quadratic_cost_keur(
                        problem.residual_demand_gwh[t] - release,
                        problem.intercept_eur_mwh[t],
                        problem.slope_eur_mwh_per_gwh[t],
                    )
                    cost += problem.spill_penalty_eur_mwh * (let_out - release)  # kEUR
                    best = min(best, cost.min() + problem.discount * value[t + 1, k2])
                value[t, k] += problem.probability[j] * best
    return value


@pytest.fixture(scope="module")
def base():
    problem = build_problem(CFG, load_weekly(PROJECT_ROOT / CFG.data.weekly_csv))
    start = time.perf_counter()
    solution = solve(problem, SDP.tol_eur_mwh, SDP.max_cycles)
    return problem, solution, time.perf_counter() - start


# --- Definition of Done (guide section 8, M4) ---------------------------------------------------


def test_one_scenario_matches_deterministic_dp() -> None:
    for problem in (toy_problem(scenarios=1), rough_problem(scenarios=1)):
        terminal = -40.0 * problem.storage_gwh  # stored water is worth 40 EUR/MWh at the end
        result = backward_pass(problem, terminal).value_keur
        reference = loop_dp(problem, terminal)
        # The reference searches a dense grid of releases, so it can only be a hair higher.
        assert (result <= reference + 1e-9).all()
        np.testing.assert_allclose(result, reference, rtol=1e-6)


def test_vectorised_stages_match_brute_force_with_spill_and_negative_prices() -> None:
    problem = rough_problem(scenarios=3)
    terminal = -20.0 * problem.storage_gwh
    result = backward_pass(problem, terminal)
    np.testing.assert_allclose(result.value_keur, loop_dp(problem, terminal), rtol=1e-6)
    assert result.spill_gwh.max() > 0  # the flood week spills


def test_water_values_are_never_below_minus_the_spill_penalty(base) -> None:
    # Spilling is always possible, so one more MWh can cost at most the spill penalty.
    for solution in (base[1], solve(rough_problem(), SDP.tol_eur_mwh, SDP.max_cycles)):
        floor = -solution.problem.spill_penalty_eur_mwh
        assert solution.water_value_eur_mwh.min() >= floor - 1e-6


def test_negative_prices_are_avoided_by_spilling() -> None:
    # From a full reservoir in the flood week the release stops where the price reaches -c_sp
    # and the rest is spilled, instead of producing at a loss up to the turbine capacity.
    problem = rough_problem(scenarios=1)
    result = backward_pass(problem, np.zeros(len(problem.storage_gwh)))
    t, full = 1, len(problem.storage_gwh) - 1
    a, b = problem.intercept_eur_mwh[t], problem.slope_eur_mwh_per_gwh[t]
    no_loss = problem.residual_demand_gwh[t] + (a + problem.spill_penalty_eur_mwh) / b
    assert no_loss < problem.release_max_gwh[t]
    assert result.release_gwh[t, full, 0] == pytest.approx(no_loss)
    assert result.spill_gwh[t, full, 0] > 0


def test_water_values_do_not_increase_with_storage(base) -> None:
    problem, solution, _ = base
    assert (np.diff(solution.water_value_eur_mwh, axis=1) <= grid_resolution_eur_mwh(problem)).all()
    # A convex toy problem (no minimum release: see the next test) is monotone as well.
    convex = SdpProblem(**{**vars(toy_problem()), "release_min_gwh": np.zeros(4)})
    toy = solve(convex, SDP.tol_eur_mwh, SDP.max_cycles)
    assert (np.diff(toy.water_value_eur_mwh, axis=1) <= grid_resolution_eur_mwh(convex)).all()


def test_production_based_minimum_release_breaks_monotonicity() -> None:
    # The finding behind reservoir.rmin_basis: with the guide's production-based R_min, output
    # that follows demand (winter) becomes a hard minimum; at low storage water is forced out at
    # this week's price and the water value rises with storage. Without R_min it does not.
    cfg = CFG.model_copy(
        update={"reservoir": CFG.reservoir.model_copy(update={"rmin_basis": "production"})}
    )
    problem = build_problem(
        cfg, load_weekly(PROJECT_ROOT / CFG.data.weekly_csv), n_storage_points=101
    )
    tolerance = grid_resolution_eur_mwh(problem)
    rise = np.diff(solve(problem, SDP.tol_eur_mwh, SDP.max_cycles).water_value_eur_mwh, axis=1)
    _, levels = np.nonzero(rise > tolerance)
    assert len(levels) > 0
    assert (problem.storage_gwh[levels + 1] <= problem.capacity_gwh / 4).all()  # low storage
    free = SdpProblem(**{**vars(problem), "release_min_gwh": np.zeros(MODEL_WEEKS_PER_YEAR)})
    rise_free = np.diff(solve(free, SDP.tol_eur_mwh, SDP.max_cycles).water_value_eur_mwh, axis=1)
    assert (rise_free <= tolerance).all()


def test_release_min_bases() -> None:
    weeks = np.tile(np.arange(1, MODEL_WEEKS_PER_YEAR + 1), 2)
    hydro = pd.Series(np.where(weeks <= 26, 1_200.0, 800.0))
    inflow = pd.Series(np.where(weeks <= 26, 400.0, 2_000.0))  # winter-like, then flood-like
    hydro.iloc[0] = np.nan  # a gap week is skipped
    production = release_min(hydro, inflow, pd.Series(weeks), 0, "production")
    capped = release_min(hydro, inflow, pd.Series(weeks), 0, "inflow_capped")
    np.testing.assert_allclose(production, np.where(np.arange(1, 53) <= 26, 1_200.0, 800.0))
    np.testing.assert_allclose(capped, np.where(np.arange(1, 53) <= 26, 400.0, 800.0))
    with pytest.raises(SdpError, match="basis"):
        release_min(hydro, inflow, pd.Series(weeks), 5, "inflow")


def test_two_identical_stages_release_the_same() -> None:
    # Two equal stages, no inflow, no discounting, no binding bounds, worthless leftover water:
    # with a convex cost the optimum splits the stored energy evenly.
    stored_gwh = 600.0
    problem = SdpProblem(
        storage_gwh=np.linspace(0.0, 2 * stored_gwh, 41),
        inflow_gwh=np.zeros((2, 1)),
        probability=np.ones(1),
        residual_demand_gwh=np.full(2, 1_000.0),
        release_min_gwh=np.zeros(2),
        release_max_gwh=np.full(2, 5_000.0),
        intercept_eur_mwh=np.full(2, 50.0),
        slope_eur_mwh_per_gwh=np.full(2, 0.05),
        spill_penalty_eur_mwh=0.01,
        discount=1.0,
    )
    result = backward_pass(problem, np.zeros(len(problem.storage_gwh)))
    start = int(np.flatnonzero(problem.storage_gwh == stored_gwh)[0])
    first = result.release_gwh[0, start, 0]
    middle = result.next_index[0, start, 0]
    second = result.release_gwh[1, middle, 0]
    assert first == pytest.approx(stored_gwh / 2)
    assert second == pytest.approx(stored_gwh / 2)
    assert result.spill_gwh[:, :, 0].max() == 0.0


def test_larger_reservoir_does_not_raise_expected_cost() -> None:
    # Same horizon and end value on both grids: the periodic solve stops when the water values
    # settle, not the level of V, so its levels are not comparable. The larger grid contains the
    # smaller one, so every policy of the small reservoir is still available.
    small = toy_problem(points=21, capacity_gwh=2_000.0)
    step = small.storage_gwh[1]
    large = toy_problem(points=31, capacity_gwh=2_000.0 + 10 * step)  # same grid step
    np.testing.assert_allclose(large.storage_gwh[:21], small.storage_gwh)
    small_value, large_value = np.zeros(21), np.zeros(31)
    for _ in range(10):  # ten years of the toy problem, from a zero end value
        small_value = backward_pass(small, small_value).value_keur[0]
        large_value = backward_pass(large, large_value).value_keur[0]
    assert (large_value[:21] <= small_value + 1e-9 * np.abs(small_value).max()).all()
    assert (large_value[:21] < small_value).any()  # the extra room is used somewhere


def test_base_case_runs_in_seconds(base) -> None:
    _, solution, seconds = base
    assert solution.converged
    assert seconds < 10.0


# --- Mechanics ----------------------------------------------------------------------------------


def test_policy_closes_the_mass_balance(base) -> None:
    problem, solution, _ = base
    s = problem.storage_gwh[None, :, None]
    inflow = problem.inflow_gwh[:, None, :]
    release, spill = solution.release_gwh, solution.spill_gwh
    np.testing.assert_allclose(solution.next_storage_gwh, s + inflow - release - spill, atol=1e-9)
    lower = np.minimum(problem.release_min_gwh[:, None, None], s + inflow)
    assert (release >= lower - 1e-9).all()
    assert (release <= problem.release_max_gwh[:, None, None] + 1e-9).all()
    assert (spill >= 0).all()
    capacity = problem.storage_gwh[-1]
    assert ((solution.next_storage_gwh >= 0) & (solution.next_storage_gwh <= capacity)).all()


def test_periodic_solution_converges(base) -> None:
    _, solution, _ = base
    history = np.array(solution.history_eur_mwh)
    assert history[-1] < SDP.tol_eur_mwh
    assert len(history) == solution.cycles - 1
    assert solution.value_keur.shape == (MODEL_WEEKS_PER_YEAR, SDP.n_storage_points)


def test_water_values_are_minus_the_value_gradient() -> None:
    grid = np.linspace(0.0, 1_000.0, 11)
    value = np.vstack([-30.0 * grid, 2e-3 * (grid - 1_000.0) ** 2])  # kEUR
    w = water_values(value, grid)  # kEUR/GWh = EUR/MWh
    np.testing.assert_allclose(w[0], 30.0)
    np.testing.assert_allclose(w[1, 1:-1], -4e-3 * (grid[1:-1] - 1_000.0))


def test_first_order_condition_holds_on_the_grid(base) -> None:
    # Interior decisions: the week's price lies between the discounted one-sided water values
    # of next week, widened by half a grid step of the price curve.
    _, solution, _ = base
    check = first_order_check(solution)
    interior = check["interior"]
    assert interior.sum() > 1_000
    price = check["price"][interior]
    assert (price >= check["lower"][interior] - 1e-6).all()
    assert (price <= check["upper"][interior] + 1e-6).all()


def test_checks_summarise_the_base_case(base) -> None:
    _, solution, _ = base
    found = checks(solution)
    assert found["rising_cells"] == 0
    assert found["interior_decisions"] > 1_000
    assert found["within_bracket_share"] == 1.0
    assert found["median_gap_eur_mwh"] < 2 * found["resolution_eur_mwh"]
    assert found["min_water_value_eur_mwh"] >= -solution.problem.spill_penalty_eur_mwh - 1e-6
    assert 1 <= found["max_rise_week"] <= MODEL_WEEKS_PER_YEAR
    assert 0 <= found["max_rise_filling_pct"] <= 100


def test_compare_solutions_tabulates_checks_and_points(base) -> None:
    _, solution, _ = base
    table = compare_solutions({"base": solution}, [(10, 10), (38, 90)])
    assert list(table.columns) == [
        "rising_cells",
        "max_rise_eur_mwh",
        "week 10 at 10%",
        "week 38 at 90%",
    ]
    assert table.loc["base", "week 10 at 10%"] == pytest.approx(water_value_at(solution, 10, 10))


def test_problem_rejects_inconsistent_inputs() -> None:
    good = toy_problem()
    with pytest.raises(SdpError, match="probabilit"):
        SdpProblem(**{**vars(good), "probability": np.array([0.5, 0.2, 0.2])})
    with pytest.raises(SdpError, match="slope"):
        SdpProblem(**{**vars(good), "slope_eur_mwh_per_gwh": np.zeros(4)})
    with pytest.raises(SdpError, match="grid"):
        SdpProblem(**{**vars(good), "storage_gwh": np.linspace(100.0, 2_000.0, 21)})
    with pytest.raises(SdpError, match="discount"):
        SdpProblem(**{**vars(good), "discount": 1.5})


def test_grid_sensitivity_compares_on_a_common_filling_grid() -> None:
    solutions = {
        points: solve(toy_problem(points=points), SDP.tol_eur_mwh, SDP.max_cycles)
        for points in (11, 21, 41)
    }
    table = grid_sensitivity(solutions, reference=41)
    assert list(table.index) == [11, 21, 41]
    assert table.loc[41, "max_abs_diff_eur_mwh"] == 0.0
    assert (table["mean_abs_diff_eur_mwh"] >= 0).all()
    assert table.loc[11, "mean_abs_diff_eur_mwh"] > table.loc[21, "mean_abs_diff_eur_mwh"]


# --- Inputs from the committed dataset and CLI --------------------------------------------------


def test_inputs_from_the_committed_dataset(base) -> None:
    problem, _, _ = base
    weekly = load_weekly(PROJECT_ROOT / CFG.data.weekly_csv)
    capacity = max(weekly["storage_gwh"].max(), weekly["storage_end_gwh"].max())
    assert problem.storage_gwh[-1] == pytest.approx(capacity)
    r_max = CFG.reservoir.hydro_capacity_mw * HOURS_PER_WEEK * CFG.reservoir.availability
    np.testing.assert_allclose(problem.release_max_gwh, r_max / MWH_PER_GWH)
    years = CFG.period.last_iso_year - CFG.period.first_iso_year + 1
    assert problem.inflow_gwh.shape == (MODEL_WEEKS_PER_YEAR, years)
    np.testing.assert_allclose(problem.probability, 1.0 / years)
    assert np.isfinite(problem.residual_demand_gwh).all()
    assert (problem.release_min_gwh <= problem.release_max_gwh).all()
    assert problem.discount == pytest.approx((1 + SDP.discount_rate_annual) ** (-1 / 52))


def test_water_values_are_high_in_late_winter_when_low_and_low_in_late_summer_when_full(
    base,
) -> None:
    # The guide's physical check, as an orientation (the owner inspects the figures).
    _, solution, _ = base
    late_winter_low = water_value_at(solution, 10, 10)
    late_summer_full = water_value_at(solution, 38, 90)
    assert late_winter_low > late_summer_full
    for week in (2, 10, 30, 44):  # within a week, more water is worth less
        assert water_value_at(solution, week, 10) > water_value_at(solution, week, 90)


def test_base_de_price_profile_takes_the_shape_without_the_crisis() -> None:
    table = price_table(load_weekly(PROJECT_ROOT / CFG.data.weekly_csv), CFG.price_model.seasons)
    profile = base_de_price_profile(CFG, table)
    assert profile.mean() == pytest.approx(de_price_profile(table).mean())  # all-years level
    calm = [y for y in range(2015, 2026) if y not in CFG.price_model.crisis_years]
    shape = de_price_profile(table, years=calm)
    np.testing.assert_allclose(profile / shape, profile.mean() / shape.mean())
    assert base_de_price_profile(CFG, table, level_eur_mwh=80.0).mean() == pytest.approx(80.0)
    levels = de_price_levels(CFG, table)
    assert levels["all_years"] == pytest.approx(profile.mean())
    assert levels["without_crisis"] == pytest.approx(shape.mean())


def _config_with(tmp_path, **sections: dict) -> str:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    for section, values in sections.items():
        raw[section].update(values)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(path)


def test_cli_solve_writes_the_outputs(tmp_path, base) -> None:
    _, solution, _ = base
    outputs = {
        "values_npz": str(tmp_path / "water_values.npz"),
        "water_values_csv": str(tmp_path / "water_values.csv"),
        "report": str(tmp_path / "sdp_report.txt"),
    }
    result = CliRunner().invoke(app, ["solve", "--config", _config_with(tmp_path, sdp=outputs)])
    assert result.exit_code == 0, result.output
    loaded = load_solution(tmp_path / "water_values.npz")
    np.testing.assert_allclose(loaded.water_value_eur_mwh, solution.water_value_eur_mwh)
    table = pd.read_csv(outputs["water_values_csv"])
    assert {"iso_week", "storage_gwh", "filling_pct", "water_value_eur_mwh"} <= set(table.columns)
    assert len(table) == MODEL_WEEKS_PER_YEAR * SDP.n_storage_points
    assert "converged" in (tmp_path / "sdp_report.txt").read_text(encoding="utf-8")
