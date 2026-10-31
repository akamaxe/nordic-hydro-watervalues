"""Forward simulation of the reservoir with the water values: historical backtest and scenarios.

Week by week, from the storage at the start of the week, the release ``R`` is chosen on a fine
grid (``simulation.decision_step_gwh``) to minimise (guide section 5.5)

    C_t(D_t - R) + c_sp Sp + delta V_{t+1}(S'),   S' = S + I_t - R - Sp,

with ``V_{t+1}`` the SDP value of the next stage interpolated linearly in storage (only its
differences matter, so the level it reached when the water values converged is irrelevant).
The release bounds are those of the SDP; water is spilled only when the reservoir would
overflow. The simulated price is ``p_t = a_t + b_t (D_t - R_t)``.

**Backtest of year y.** Start from the observed storage at the start of the year and use the
year's observed inflow, residual demand and German price. The water values are recomputed for
that year with the same residual demand and German price (perfect foresight of both, so the
only uncertainty is hydrological), with the price curve refitted without the year
(leave-one-year-out) and with the year left out of the inflow scenarios, so that neither the
price curve nor the inflow distribution has seen it. The minimum release and ``S_max`` still
come from all years (one year of eleven; its effect was checked to be negligible). Weeks
without an observed residual demand use the climatological one; a missing German price takes
the base seasonal shape at the year's own level; both are flagged and left out of the price
metrics.

**Scenarios** (guide section 5.6): the German price profile scaled to several annual means,
and the driest against the wettest year of inflow from the same initial storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from watervalues.config import PROJECT_ROOT, AppConfig
from watervalues.inflow import IMPUTED, InflowResult, build_inflow, dry_and_wet
from watervalues.price import base_spec, fit_curve, price_table, quadratic_cost_keur, without_year
from watervalues.sdp import (
    SdpProblem,
    SdpSolution,
    base_de_price_profile,
    build_problem,
    capacity_gwh,
    checks,
    solve,
)
from watervalues.units import EUR_PER_KEUR, HOURS_PER_WEEK, MODEL_WEEKS_PER_YEAR, MWH_PER_GWH


class SimulationError(RuntimeError):
    """The simulation inputs are inconsistent."""


def simulate(
    solution: SdpSolution,
    inflow_gwh: np.ndarray,
    residual_demand_gwh: np.ndarray,
    intercept_eur_mwh: np.ndarray,
    slope_eur_mwh_per_gwh: np.ndarray,
    start_storage_gwh: float,
    decision_step_gwh: float,
) -> pd.DataFrame:
    """Operate the reservoir for one year with the water values of an SDP solution.

    Parameters
    ----------
    solution : SdpSolution
        Water values (its value function) and the release bounds, spill penalty and discount.
    inflow_gwh, residual_demand_gwh : numpy.ndarray
        Inflow and residual demand of each stage ``(52,)`` [GWh per 168-hour stage].
    intercept_eur_mwh, slope_eur_mwh_per_gwh : numpy.ndarray
        Weekly price curve ``p_t(N) = a_t + b_t N`` ``(52,)`` [EUR/MWh, EUR/MWh per GWh].
    start_storage_gwh : float
        Storage at the start of the first week [GWh], within ``[0, S_max]``.
    decision_step_gwh : float
        Step of the release grid [GWh].

    Returns
    -------
    pandas.DataFrame
        One row per ISO week: ``storage_gwh`` (start), ``inflow_gwh``, ``release_gwh``,
        ``spill_gwh``, ``storage_end_gwh``, ``net_import_gwh`` (``D_t - R_t``) [GWh] and
        ``price_eur_mwh`` [EUR/MWh]. ``storage_end_gwh = storage_gwh + inflow_gwh -
        release_gwh - spill_gwh`` holds exactly.

    Raises
    ------
    SimulationError
        If an input does not have one finite value per stage, the inflow is negative, or the
        start storage is outside the reservoir.
    """
    p = solution.problem
    stages = p.stages
    series = (inflow_gwh, residual_demand_gwh, intercept_eur_mwh, slope_eur_mwh_per_gwh)
    arrays = [np.asarray(values, dtype=float) for values in series]
    if any(a.shape != (stages,) or not np.isfinite(a).all() for a in arrays):
        raise SimulationError(f"inflow, demand and price curve need {stages} finite values")
    if (arrays[0] < 0).any():
        raise SimulationError("inflow must be >= 0 in every stage")
    if not 0 <= start_storage_gwh <= p.capacity_gwh:
        raise SimulationError(f"start storage {start_storage_gwh} GWh is outside the reservoir")
    inflow, demand, intercept, slope = arrays
    rows = []
    storage = float(start_storage_gwh)
    for t in range(stages):
        available = storage + inflow[t]
        lower = min(p.release_min_gwh[t], available)
        upper = min(p.release_max_gwh[t], available)
        release = np.append(np.arange(lower, upper, decision_step_gwh), upper)
        kept = available - release
        spill = np.maximum(kept - p.capacity_gwh, 0.0)
        storage_end = kept - spill
        value_next = solution.value_after(t)  # after week 52 comes week 1
        cost = (
            quadratic_cost_keur(demand[t] - release, intercept[t], slope[t])
            + p.spill_penalty_eur_mwh * spill * MWH_PER_GWH / EUR_PER_KEUR
            + p.discount * np.interp(storage_end, p.storage_gwh, value_next)
        )
        best = int(np.argmin(cost))
        rows.append(
            {
                "iso_week": t + 1,
                "storage_gwh": storage,
                "inflow_gwh": inflow[t],
                "release_gwh": release[best],
                "spill_gwh": spill[best],
                "storage_end_gwh": storage_end[best],
                "net_import_gwh": demand[t] - release[best],
                "price_eur_mwh": intercept[t] + slope[t] * (demand[t] - release[best]),
            }
        )
        storage = float(storage_end[best])
    return pd.DataFrame(rows)


def _fill(observed: pd.Series, fallback: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Observed weekly values with gaps filled from a fallback, and the mask of filled weeks."""
    values = observed.to_numpy(dtype=float)
    missing = np.isnan(values)
    return np.where(missing, fallback, values), missing


@dataclass(frozen=True, eq=False)
class BacktestYear:
    """One simulated year: its weekly trace and the checks of the water values behind it."""

    trace: pd.DataFrame
    checks: dict[str, float]


def solution_checks(solution: SdpSolution) -> dict[str, float]:
    """Convergence and :func:`watervalues.sdp.checks` of one SDP solution, as one record."""
    return {"converged": float(solution.converged), "cycles": float(solution.cycles)} | checks(
        solution
    )


def backtest_year(
    cfg: AppConfig,
    weekly: pd.DataFrame,
    year: int,
    *,
    table: pd.DataFrame | None = None,
    inflow: InflowResult | None = None,
    climatology: SdpProblem | None = None,
) -> BacktestYear:
    """Simulate one historical year with a price curve and inflow scenarios that have not seen it.

    Parameters
    ----------
    cfg : AppConfig
        Configuration.
    weekly : pandas.DataFrame
        Weekly dataset (``weekly_se.csv`` contract) covering ``cfg.period``.
    year : int
        ISO year to simulate.
    table, inflow : pandas.DataFrame, InflowResult or None
        Price table and inflow of ``weekly`` (computed when None; pass them to reuse them).
    climatology : SdpProblem or None
        The base-case problem, whose residual demand fills the weeks without an observed one
        (built when None).

    Returns
    -------
    BacktestYear
        ``trace``: the columns of :func:`simulate` plus ``iso_year``, ``week_start``,
        ``observed_storage_gwh`` and ``observed_storage_end_gwh`` [GWh],
        ``observed_price_eur_mwh`` [EUR/MWh], ``observed_net_import_gwh`` [GWh per stage],
        ``curve_price_eur_mwh`` (the year's price curve at the observed net import) [EUR/MWh],
        ``filled`` (residual demand, German price or inflow came from the climatology) and
        ``folded`` (a week 52 that also holds ISO week 53, whose observed end storage is a
        week later than the simulated one). ``checks``: :func:`solution_checks` of the year's
        water values.

    Raises
    ------
    SimulationError
        If the year is incomplete or its water values did not converge.
    """
    pm = cfg.price_model
    if table is None:
        table = price_table(weekly, pm.seasons)
    if inflow is None:
        inflow = build_inflow(cfg, weekly)
    if climatology is None:
        climatology = build_problem(cfg, weekly, table=table, inflow=inflow)
    rows = (table["iso_year"] == year).to_numpy()
    if rows.sum() != MODEL_WEEKS_PER_YEAR:
        raise SimulationError(f"the weekly dataset has no complete ISO year {year}")
    years = sorted(int(y) for y in table["iso_year"].unique())
    curve = fit_curve(table, without_year(base_spec(cfg), year), pm.hac_maxlags)
    observed_de = table.loc[rows, "price_de_eur_mwh"]
    # Gaps take the base seasonal shape at the year's own price level.
    fallback_de = base_de_price_profile(cfg, table, level_eur_mwh=float(observed_de.mean()))
    price_de, de_filled = _fill(observed_de, fallback_de)
    demand, demand_filled = _fill(
        table.loc[rows, "residual_demand_stage_gwh"], climatology.residual_demand_gwh
    )
    problem = build_problem(
        cfg,
        weekly,
        de_price_eur_mwh=price_de,
        residual_demand_gwh=demand,
        curve=curve,
        inflow_years=[y for y in years if y != year],
        table=table,
        inflow=inflow,
    )
    solution = solve(problem, cfg.sdp.tol_eur_mwh, cfg.sdp.max_cycles)
    if not solution.converged:
        raise SimulationError(f"the water values of {year} did not converge")
    stage = curve.stage_curve(price_de)
    observed = weekly.loc[rows]
    trace = simulate(
        solution,
        inflow.weekly.loc[rows, "inflow_stage_gwh"].to_numpy(),
        demand,
        stage.intercept_eur_mwh,
        stage.slope_eur_mwh_per_gwh,
        float(observed["storage_gwh"].iloc[0]),
        cfg.simulation.decision_step_gwh,
    )
    observed_import = table.loc[rows, "net_import_stage_gwh"].to_numpy()
    trace.insert(0, "iso_year", year)
    trace.insert(1, "week_start", observed["week_start"].to_numpy())
    trace["observed_storage_gwh"] = observed["storage_gwh"].to_numpy()
    trace["observed_storage_end_gwh"] = observed["storage_end_gwh"].to_numpy()
    trace["observed_price_eur_mwh"] = observed["price_se_eur_mwh"].to_numpy()
    trace["observed_net_import_gwh"] = observed_import
    trace["curve_price_eur_mwh"] = (
        stage.intercept_eur_mwh + stage.slope_eur_mwh_per_gwh * observed_import
    )
    imputed = (inflow.weekly.loc[rows, "status"] == IMPUTED).to_numpy()
    trace["filled"] = de_filled | demand_filled | imputed
    trace["folded"] = (observed["hours"] != HOURS_PER_WEEK).to_numpy()
    return BacktestYear(trace, solution_checks(solution))


def _correlation(simulated: pd.Series, observed: pd.Series) -> float:
    """Pearson correlation [-]; NaN with fewer than two pairs."""
    return float(simulated.corr(observed)) if len(simulated) > 1 else float("nan")


def backtest_metrics(traces: pd.DataFrame, capacity_gwh: float) -> pd.DataFrame:
    """Accuracy of the backtest per year and over all years.

    The price metrics use the weeks with an observed price and no filled input. The price bias
    splits into ``curve_bias`` (the year's curve at the observed net import minus the observed
    price: what the price model misses) and ``operating_bias`` (simulated price minus that
    curve price: the effect of operating differently from reality). The storage error compares
    the simulated and observed storage at the end of every week except a folded week 52.

    Parameters
    ----------
    traces : pandas.DataFrame
        The traces of :func:`backtest_year` for one or more years.
    capacity_gwh : float
        ``S_max`` [GWh], the 100 % of the storage error.

    Returns
    -------
    pandas.DataFrame
        Indexed by ISO year plus ``all``: ``price_corr`` [-], ``price_mae_eur_mwh``,
        ``price_bias_eur_mwh``, ``curve_bias_eur_mwh``, ``operating_bias_eur_mwh``,
        ``storage_rmse_pct`` [% of S_max], ``spill_gwh`` [GWh], ``observed_price_mean_eur_mwh``
        and ``simulated_price_mean_eur_mwh`` [EUR/MWh]. The ``all`` row also has
        ``price_corr_within_year`` (prices less each year's mean) and
        ``price_corr_annual_means`` (the yearly means).
    """

    def metrics(frame: pd.DataFrame) -> dict[str, float]:
        priced = frame[~frame["filled"] & frame["observed_price_eur_mwh"].notna()]
        error = priced["price_eur_mwh"] - priced["observed_price_eur_mwh"]
        curve = priced.dropna(subset=["curve_price_eur_mwh"])
        compared = frame[~frame["folded"]]
        storage_error = compared["storage_end_gwh"] - compared["observed_storage_end_gwh"]
        return {
            "price_corr": _correlation(priced["price_eur_mwh"], priced["observed_price_eur_mwh"]),
            "price_mae_eur_mwh": float(error.abs().mean()),
            "price_bias_eur_mwh": float(error.mean()),
            "curve_bias_eur_mwh": float(
                (curve["curve_price_eur_mwh"] - curve["observed_price_eur_mwh"]).mean()
            ),
            "operating_bias_eur_mwh": float(
                (curve["price_eur_mwh"] - curve["curve_price_eur_mwh"]).mean()
            ),
            "storage_rmse_pct": float(100 * np.sqrt((storage_error**2).mean()) / capacity_gwh),
            "spill_gwh": float(frame["spill_gwh"].sum()),
            "observed_price_mean_eur_mwh": float(priced["observed_price_eur_mwh"].mean()),
            "simulated_price_mean_eur_mwh": float(priced["price_eur_mwh"].mean()),
        }

    rows = {int(year): metrics(frame) for year, frame in traces.groupby("iso_year")}
    pooled = metrics(traces)
    priced = traces[~traces["filled"] & traces["observed_price_eur_mwh"].notna()]
    by_year = priced.groupby("iso_year")
    within = priced[["price_eur_mwh", "observed_price_eur_mwh"]] - by_year[
        ["price_eur_mwh", "observed_price_eur_mwh"]
    ].transform("mean")
    means = by_year[["price_eur_mwh", "observed_price_eur_mwh"]].mean()
    pooled["price_corr_within_year"] = _correlation(
        within["price_eur_mwh"], within["observed_price_eur_mwh"]
    )
    pooled["price_corr_annual_means"] = _correlation(
        means["price_eur_mwh"], means["observed_price_eur_mwh"]
    )
    rows["all"] = pooled
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "iso_year"
    return table


def water_value_on_path(solution: SdpSolution, filling_pct: np.ndarray) -> np.ndarray:
    """Water value of each stage at a given storage level per stage [EUR/MWh].

    Parameters
    ----------
    solution : SdpSolution
        Water values ``(52, K)``.
    filling_pct : numpy.ndarray
        Storage of each stage ``(52,)`` [% of ``S_max``].

    Returns
    -------
    numpy.ndarray
        Shape ``(52,)``.
    """
    s = solution.problem.storage_gwh
    return np.array(
        [
            np.interp(pct / 100 * s[-1], s, row)
            for pct, row in zip(filling_pct, solution.water_value_eur_mwh, strict=True)
        ]
    )


def mean_filling_pct(cfg: AppConfig, weekly: pd.DataFrame) -> np.ndarray:
    """Mean observed storage at the start of each ISO week [% of ``S_max``], shape ``(52,)``."""
    capacity = capacity_gwh(cfg, weekly)
    weeks = range(1, MODEL_WEEKS_PER_YEAR + 1)
    mean = weekly.groupby("iso_week")["storage_gwh"].mean().reindex(weeks)
    return (100 * mean / capacity).to_numpy()


def de_price_scenarios(
    cfg: AppConfig,
    weekly: pd.DataFrame,
    *,
    table: pd.DataFrame | None = None,
    inflow: InflowResult | None = None,
) -> tuple[pd.DataFrame, float, dict[float, SdpSolution]]:
    """Water values for the German price profile scaled to each scenario level.

    Parameters
    ----------
    cfg : AppConfig
        Configuration (``scenarios.de_price_levels_eur_mwh``).
    weekly : pandas.DataFrame
        Weekly dataset.
    table, inflow : pandas.DataFrame, InflowResult or None
        Price table and inflow of ``weekly`` (computed when None).

    Returns
    -------
    tuple of (pandas.DataFrame, float, dict)
        Indexed by the level [EUR/MWh]: ``water_value_mean_eur_mwh`` (mean over the year of the
        water value at the mean observed storage of each week), its ``min`` and ``max`` over
        the weeks, and ``price_mean_eur_mwh`` (mean of ``a_t``, the price at zero net import);
        the slope of the mean water value against the level [EUR/MWh per EUR/MWh]; and the
        solution of each level.
    """
    if table is None:
        table = price_table(weekly, cfg.price_model.seasons)
    if inflow is None:
        inflow = build_inflow(cfg, weekly)
    path = mean_filling_pct(cfg, weekly)
    rows, solutions = {}, {}
    for level in cfg.scenarios.de_price_levels_eur_mwh:
        profile = base_de_price_profile(cfg, table, level)
        problem = build_problem(cfg, weekly, de_price_eur_mwh=profile, table=table, inflow=inflow)
        solution = solve(problem, cfg.sdp.tol_eur_mwh, cfg.sdp.max_cycles)
        w = water_value_on_path(solution, path)
        rows[float(level)] = {
            "water_value_mean_eur_mwh": float(w.mean()),
            "water_value_min_eur_mwh": float(w.min()),
            "water_value_max_eur_mwh": float(w.max()),
            "price_mean_eur_mwh": float(problem.intercept_eur_mwh.mean()),
        }
        solutions[float(level)] = solution
    frame = pd.DataFrame.from_dict(rows, orient="index").rename_axis("de_price_level_eur_mwh")
    slope = float(np.polyfit(frame.index, frame["water_value_mean_eur_mwh"], 1)[0])
    return frame, slope, solutions


def dry_and_wet_years(
    cfg: AppConfig,
    weekly: pd.DataFrame,
    solution: SdpSolution,
    *,
    inflow: InflowResult | None = None,
) -> pd.DataFrame:
    """Simulate the driest and the wettest inflow year from the same initial storage.

    Both use the base-case water values, the climatological residual demand and the base-case
    German price profile, and start from the mean observed storage of ISO week 1.

    Parameters
    ----------
    cfg : AppConfig
        Configuration.
    weekly : pandas.DataFrame
        Weekly dataset.
    solution : SdpSolution
        Base-case solution (its problem supplies ``D_t``, ``a_t`` and ``b_t``).
    inflow : InflowResult or None
        Inflow of ``weekly`` (computed when None).

    Returns
    -------
    pandas.DataFrame
        The columns of :func:`simulate` plus ``case`` (``dry`` or ``wet``) and ``iso_year``
        (the inflow year).
    """
    if inflow is None:
        inflow = build_inflow(cfg, weekly)
    driest, wettest = dry_and_wet(inflow.annual)
    p = solution.problem
    start = mean_filling_pct(cfg, weekly)[0] / 100 * p.capacity_gwh
    traces = []
    for case, year in (("dry", driest), ("wet", wettest)):
        trace = simulate(
            solution,
            inflow.scenarios.loc[year].to_numpy(),
            p.residual_demand_gwh,
            p.intercept_eur_mwh,
            p.slope_eur_mwh_per_gwh,
            start,
            cfg.simulation.decision_step_gwh,
        )
        trace.insert(0, "case", case)
        trace.insert(1, "iso_year", year)
        traces.append(trace)
    return pd.concat(traces, ignore_index=True)


@dataclass(frozen=True, eq=False)
class SimulationResult:
    """Backtest traces and metrics, the scenarios, the checks of every SDP and the report."""

    backtest: pd.DataFrame
    metrics: pd.DataFrame
    de_price: pd.DataFrame
    de_price_slope: float
    dry_wet: pd.DataFrame
    checks: pd.DataFrame
    report: list[str]


def _checks_lines(found: pd.DataFrame) -> list[str]:
    rising = found[found["rising_cells"] > 0]
    lines = [
        "\nChecks of the water values behind every simulation (converged; water values "
        "non-increasing in storage within the grid resolution, see watervalues solve)",
        found[
            [
                "converged",
                "cycles",
                "rising_cells",
                "max_rise_eur_mwh",
                "resolution_eur_mwh",
                "max_rise_week",
                "max_rise_filling_pct",
                "rising_max_filling_pct",
                "min_water_value_eur_mwh",
            ]
        ]
        .round(2)
        .to_string(),
    ]
    if rising.empty:
        lines.append("  every solution passes")
    else:
        lines.append(
            f"  water values rise with storage beyond the grid resolution in {len(rising)} of "
            f"{len(found)} solutions, all at or below "
            f"{rising['rising_max_filling_pct'].max():.1f}% filling: the kink of the minimum "
            "release at a near-empty reservoir (forced release of all the water)"
        )
    return lines


def simulation_report(cfg: AppConfig, result: SimulationResult) -> list[str]:
    """Human-readable summary of the backtest, the scenarios and the checks.

    Parameters
    ----------
    cfg : AppConfig
        Configuration.
    result : SimulationResult
        The results (its ``report`` is ignored).

    Returns
    -------
    list of str
    """
    m = result.metrics
    filled = result.backtest[result.backtest["filled"]]
    weeks = ", ".join(
        f"{y}-W{k:02d}" for y, k in zip(filled["iso_year"], filled["iso_week"], strict=True)
    )
    lines = [
        "Backtest: each ISO year simulated from its observed starting storage with its observed "
        "inflow, residual demand and German price. Its water values use the same residual "
        "demand and German price (so the only uncertainty is hydrological), a price curve fitted "
        "without the year and inflow scenarios without the year",
        f"  release grid {cfg.simulation.decision_step_gwh:g} GWh; weeks with a filled residual "
        f"demand, German price or inflow (left out of the price metrics): {weeks or 'none'}",
        "  a folded week 52 (with ISO week 53) is left out of the storage error",
        "\nMetrics per year: weekly price correlation, MAE and bias (simulated - observed) "
        "[EUR/MWh], split into the price curve's error at the observed net import and the "
        "effect of operating differently; storage RMSE at the end of each week [% of S_max]; "
        "spill [GWh]",
        m.round(2).to_string(),
    ]
    lines += _checks_lines(result.checks)
    lines += [
        "\nScenario: German price profile scaled to each annual mean; water value at the mean "
        "observed storage of each week [EUR/MWh]",
        result.de_price.round(1).to_string(),
        f"  slope of the mean water value: {result.de_price_slope:.2f} EUR/MWh per EUR/MWh of "
        "German price",
        "\nScenario: driest against wettest inflow year from the same initial storage",
    ]
    for case, trace in result.dry_wet.groupby("case", sort=False):
        lines.append(
            f"  {case} ({int(trace['iso_year'].iloc[0])}): inflow {trace['inflow_gwh'].sum():,.0f} "
            f"GWh, release {trace['release_gwh'].sum():,.0f} GWh, spill "
            f"{trace['spill_gwh'].sum():,.0f} GWh; storage {trace['storage_end_gwh'].min():,.0f} "
            f"to {trace['storage_end_gwh'].max():,.0f} GWh, ending at "
            f"{trace['storage_end_gwh'].iloc[-1]:,.0f}; price mean "
            f"{trace['price_eur_mwh'].mean():.1f} EUR/MWh (weeks "
            f"{trace['price_eur_mwh'].min():.1f} to {trace['price_eur_mwh'].max():.1f})"
        )
    return lines


def run_simulations(
    cfg: AppConfig, weekly: pd.DataFrame, years: list[int] | None = None
) -> SimulationResult:
    """Backtest every year (or the given ones), run the scenarios and check every SDP.

    Parameters
    ----------
    cfg : AppConfig
        Configuration.
    weekly : pandas.DataFrame
        Weekly dataset covering ``cfg.period``.
    years : list of int or None
        ISO years to backtest; all years of the period when None.

    Returns
    -------
    SimulationResult

    Raises
    ------
    SimulationError
        If the water values of a year or a scenario did not converge.
    """
    table = price_table(weekly, cfg.price_model.seasons)
    inflow = build_inflow(cfg, weekly)
    climatology = build_problem(cfg, weekly, table=table, inflow=inflow)
    if years is None:
        years = list(range(cfg.period.first_iso_year, cfg.period.last_iso_year + 1))
    runs = {
        y: backtest_year(cfg, weekly, y, table=table, inflow=inflow, climatology=climatology)
        for y in years
    }
    traces = pd.concat([run.trace for run in runs.values()], ignore_index=True)
    metrics = backtest_metrics(traces, capacity_gwh(cfg, weekly))
    de_price, slope, scenario_solutions = de_price_scenarios(
        cfg, weekly, table=table, inflow=inflow
    )
    base = solve(climatology, cfg.sdp.tol_eur_mwh, cfg.sdp.max_cycles)
    unconverged = [
        f"{level:g} EUR/MWh" for level, s in scenario_solutions.items() if not s.converged
    ]
    if not base.converged or unconverged:
        raise SimulationError(f"water values did not converge: {unconverged or ['base case']}")
    dry_wet = dry_and_wet_years(cfg, weekly, base, inflow=inflow)
    records = {f"backtest {y}": run.checks for y, run in runs.items()}
    records |= {
        f"German price {level:g} EUR/MWh": solution_checks(s)
        for level, s in scenario_solutions.items()
    }
    records["base case (dry and wet years)"] = solution_checks(base)
    found = pd.DataFrame.from_dict(records, orient="index").rename_axis("solution")
    result = SimulationResult(traces, metrics, de_price, slope, dry_wet, found, [])
    return SimulationResult(
        traces, metrics, de_price, slope, dry_wet, found, simulation_report(cfg, result)
    )


def write_outputs(cfg: AppConfig, result: SimulationResult) -> None:
    """Write the backtest, its metrics, the scenarios, the checks and the report."""
    paths = cfg.simulation
    float_format = cfg.data.csv_float_format
    outputs = {
        paths.backtest_csv: result.backtest,
        paths.metrics_csv: result.metrics,
        paths.de_price_csv: result.de_price,
        paths.dry_wet_csv: result.dry_wet,
        paths.checks_csv: result.checks,
    }
    for relative, frame in outputs.items():
        path = PROJECT_ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        index = frame.index.name is not None
        frame.to_csv(path, index=index, float_format=float_format, lineterminator="\n")
    report_path = PROJECT_ROOT / paths.report
    report_path.write_text("\n".join(result.report) + "\n", encoding="utf-8", newline="\n")


def load_metrics(path: Path) -> pd.DataFrame:
    """Read the backtest metrics written by :func:`write_outputs`."""
    return pd.read_csv(path, index_col="iso_year", encoding="utf-8")
