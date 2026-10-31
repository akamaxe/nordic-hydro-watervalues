"""Water values of the aggregated Swedish reservoir by stochastic dynamic programming (SDP).

The reservoir has one state, the stored energy ``S`` [GWh] at the start of a weekly stage. For
stage ``t`` and inflow scenario ``j`` (probability ``pi_j``), the expected cost-to-go is
(guide section 5.3)

    V_t(S) = sum_j pi_j min_{R, Sp} [ C_t(D_t - R) + c_sp Sp + delta V_{t+1}(S') ],

    S' = S + I_tj - R - Sp,   0 <= S' <= S_max,   min(R_min_t, S + I_tj) <= R <= R_max_t,   Sp >= 0,

with ``C_t`` the cost of covering net import ``D_t - R`` on the price curve of
:mod:`watervalues.price` [kEUR], ``D_t`` the climatological residual demand, ``R`` the hydro
release and ``Sp`` the spill [GWh per stage], ``c_sp`` a small spill penalty and ``delta`` the
weekly discount factor. Inflow is known when the week's decision is made (hazard-decision) and
independent between weeks.

The water value is ``w_t(S) = -dV_t/dS`` [kEUR/GWh = EUR/MWh], by centred differences on the
grid. At an interior optimum the week's price equals the discounted water value of next week,
``p_t(D_t - R*) = delta w_{t+1}(S')``.

Algorithm (guide section 5.4): ``K`` uniform storage points; backward over the stages with the
candidates ``S'`` on the same grid, vectorised over ``(S, scenario, S')`` (loop over stages only);
the year is repeated with the value of stage 1 as the end value until no water value moves more
than the tolerance between cycles. With ``delta < 1`` the cycle is a contraction and converges.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

from watervalues.config import PROJECT_ROOT, AppConfig
from watervalues.inflow import build_inflow
from watervalues.price import (
    PriceCurve,
    base_spec,
    de_price_profile,
    fit_curve,
    price_table,
    quadratic_cost_keur,
)
from watervalues.units import (
    EUR_PER_KEUR,
    HOURS_PER_WEEK,
    MODEL_WEEKS_PER_YEAR,
    MWH_PER_GWH,
)

_PROBABILITY_TOL = 1e-9  # numerical tolerance of the sum of probabilities
_GRID_RTOL = 1e-9  # numerical tolerance of a uniform grid step
_VALUE_TOL = 1e-6  # numerical tolerance of the lowest water value [EUR/MWh]
_ARRAYS = (
    "storage_gwh",
    "inflow_gwh",
    "probability",
    "residual_demand_gwh",
    "release_min_gwh",
    "release_max_gwh",
    "intercept_eur_mwh",
    "slope_eur_mwh_per_gwh",
)


class SdpError(RuntimeError):
    """The SDP inputs are inconsistent or the solution failed a check."""


@dataclass(frozen=True, eq=False)
class SdpProblem:
    """Inputs of the SDP; arrays are copied and read-only.

    Attributes
    ----------
    storage_gwh : numpy.ndarray
        Uniform storage grid ``(K,)`` [GWh] from 0 to ``S_max``.
    inflow_gwh : numpy.ndarray
        Inflow ``(T, J)`` per stage and scenario [GWh per 168-hour stage].
    probability : numpy.ndarray
        Scenario probabilities ``(J,)``, summing to 1.
    residual_demand_gwh : numpy.ndarray
        ``D_t`` ``(T,)`` [GWh per stage].
    release_min_gwh, release_max_gwh : numpy.ndarray
        Release bounds ``(T,)`` [GWh per stage]; the lower one gives way when the water is short.
    intercept_eur_mwh, slope_eur_mwh_per_gwh : numpy.ndarray
        Weekly price curve ``p_t(N) = a_t + b_t N`` ``(T,)`` [EUR/MWh, EUR/MWh per GWh];
        ``b_t > 0``.
    spill_penalty_eur_mwh : float
        ``c_sp`` [EUR/MWh].
    discount : float
        Discount factor per stage, in (0, 1].
    """

    storage_gwh: np.ndarray
    inflow_gwh: np.ndarray
    probability: np.ndarray
    residual_demand_gwh: np.ndarray
    release_min_gwh: np.ndarray
    release_max_gwh: np.ndarray
    intercept_eur_mwh: np.ndarray
    slope_eur_mwh_per_gwh: np.ndarray
    spill_penalty_eur_mwh: float
    discount: float

    def __post_init__(self) -> None:
        for name in _ARRAYS:
            values = np.array(getattr(self, name), dtype=float)
            if not np.isfinite(values).all():
                raise SdpError(f"{name} has missing or infinite values")
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        s = self.storage_gwh
        steps = np.diff(s)
        if s.ndim != 1 or len(s) < 2 or s[0] != 0 or (steps <= 0).any():  # noqa: PLR2004
            raise SdpError("the storage grid must start at 0 GWh and increase strictly")
        if not np.allclose(steps, steps[0], rtol=_GRID_RTOL, atol=0):
            raise SdpError("the storage grid must be uniform")
        if self.inflow_gwh.ndim != 2 or (self.inflow_gwh < 0).any():  # noqa: PLR2004
            raise SdpError("inflow must be a (stages, scenarios) array of values >= 0")
        stages, scenarios = self.inflow_gwh.shape
        if self.probability.shape != (scenarios,) or (self.probability < 0).any():
            raise SdpError(f"probabilities must be {scenarios} values >= 0")
        if abs(self.probability.sum() - 1) > _PROBABILITY_TOL:
            raise SdpError(f"probabilities sum to {self.probability.sum()}, not 1")
        for name in _ARRAYS[3:]:
            if getattr(self, name).shape != (stages,):
                raise SdpError(f"{name} needs one value per stage ({stages})")
        if (self.release_min_gwh < 0).any() or (self.release_min_gwh > self.release_max_gwh).any():
            raise SdpError("release bounds must satisfy 0 <= release_min <= release_max")
        if (self.slope_eur_mwh_per_gwh <= 0).any():
            raise SdpError("the price slope must be > 0 in every stage (convex cost)")
        if self.spill_penalty_eur_mwh < 0:
            raise SdpError("the spill penalty must be >= 0")
        if not 0 < self.discount <= 1:
            raise SdpError(f"the discount factor must be in (0, 1]: {self.discount}")

    @property
    def stages(self) -> int:
        """Number of stages ``T``."""
        return self.inflow_gwh.shape[0]

    @property
    def capacity_gwh(self) -> float:
        """``S_max`` [GWh]."""
        return float(self.storage_gwh[-1])


@dataclass(frozen=True, eq=False)
class BackwardPass:
    """One backward pass over the stages.

    ``value_keur`` ``(T + 1, K)`` is the expected cost-to-go at the start of each stage, the
    last row being the end value [kEUR]. ``next_index`` ``(T, K, J)`` is the optimal next
    storage as a grid index, with the matching ``release_gwh`` and ``spill_gwh`` [GWh].
    """

    value_keur: np.ndarray
    next_index: np.ndarray
    release_gwh: np.ndarray
    spill_gwh: np.ndarray


def _stage(
    problem: SdpProblem, t: int, value_next_keur: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Optimal decisions of stage ``t`` for every storage level and scenario.

    On a uniform grid the water let out, ``X = R + Sp = S_k + I_j - S_k'``, depends only on
    ``d = k - k'`` and the scenario. The immediate cost is therefore computed once per
    ``(d, j)`` and gathered into the ``(K, J, K')`` array of candidates (storage now,
    scenario, storage next), to which the discounted value of ``S'`` is added.

    Release and spill are both decisions. For a given ``X`` the cost
    ``C_t(D_t - R) + c_sp (X - R)`` is convex in ``R`` and smallest where the price equals
    ``-c_sp``, at ``R = D_t + (a_t + c_sp) / b_t``; the optimal release is that point clipped
    to ``[R_min_t, min(X, R_max)]``. While prices are positive this is ``min(X, R_max)``: spill
    only above turbine capacity. When the price curve would go below ``-c_sp``, spilling the
    rest is cheaper than producing it.
    """
    s = problem.storage_gwh
    points = len(s)
    scenario = np.arange(problem.inflow_gwh.shape[1])
    released = (
        np.arange(1 - points, points)[:, None] * (s[1] - s[0]) + problem.inflow_gwh[t][None, :]
    )  # (2K - 1, J): R + Sp for d = k - k' [GWh]
    # Release at which one more MWh would sell below -c_sp, then clipped to the bounds; with
    # X < R_min (only allowed for S' = 0, below) this gives R = X: everything is produced.
    no_loss = (
        problem.residual_demand_gwh[t]
        + (problem.intercept_eur_mwh[t] + problem.spill_penalty_eur_mwh)
        / problem.slope_eur_mwh_per_gwh[t]
    )
    release = np.minimum(
        np.minimum(released, problem.release_max_gwh[t]),
        max(no_loss, problem.release_min_gwh[t]),
    )
    spill = released - release
    cost = quadratic_cost_keur(
        problem.residual_demand_gwh[t] - release,
        problem.intercept_eur_mwh[t],
        problem.slope_eur_mwh_per_gwh[t],
    )
    cost += problem.spill_penalty_eur_mwh * spill * MWH_PER_GWH / EUR_PER_KEUR
    # Feasible: at least R_min released, or all the water released (S' = 0, below).
    immediate = np.where(released >= problem.release_min_gwh[t], cost, np.inf)
    d = np.arange(points)[:, None] - np.arange(points)[None, :] + points - 1  # (K, K')
    total = immediate[d[:, None, :], scenario[None, :, None]]  # (K, J, K')
    total[:, :, 0] = cost[d[:, 0]]  # S' = 0 releases everything: always allowed
    total += problem.discount * value_next_keur[None, None, :]
    index = total.argmin(axis=2)  # (K, J)
    best = np.take_along_axis(total, index[:, :, None], axis=2)[:, :, 0]
    chosen = d[np.arange(points)[:, None], index]  # d of the optimum, (K, J)
    return (
        best @ problem.probability,
        index,
        release[chosen, scenario[None, :]],
        spill[chosen, scenario[None, :]],
    )


def backward_pass(problem: SdpProblem, end_value_keur: np.ndarray) -> BackwardPass:
    """Solve the stages backwards from an end value.

    Parameters
    ----------
    problem : SdpProblem
        The inputs.
    end_value_keur : numpy.ndarray
        Value after the last stage on the storage grid ``(K,)`` [kEUR].

    Returns
    -------
    BackwardPass
    """
    stages, scenarios = problem.inflow_gwh.shape
    points = len(problem.storage_gwh)
    value = np.empty((stages + 1, points))
    value[stages] = end_value_keur
    next_index = np.empty((stages, points, scenarios), dtype=np.intp)
    release = np.empty((stages, points, scenarios))
    spill = np.empty((stages, points, scenarios))
    for t in reversed(range(stages)):
        value[t], next_index[t], release[t], spill[t] = _stage(problem, t, value[t + 1])
    return BackwardPass(value, next_index, release, spill)


def water_values(value_keur: np.ndarray, storage_gwh: np.ndarray) -> np.ndarray:
    """Water values ``w = -dV/dS`` by centred differences (one-sided at the grid ends).

    Parameters
    ----------
    value_keur : numpy.ndarray
        Expected cost-to-go ``(..., K)`` [kEUR].
    storage_gwh : numpy.ndarray
        Storage grid ``(K,)`` [GWh].

    Returns
    -------
    numpy.ndarray
        Water values, shaped like ``value_keur`` [kEUR/GWh = EUR/MWh].
    """
    return -np.gradient(value_keur, storage_gwh, axis=-1)


@dataclass(frozen=True, eq=False)
class SdpSolution:
    """Periodic solution of the last cycle.

    ``value_keur`` and ``water_value_eur_mwh`` are ``(T, K)`` at the start of stages 1-T
    [kEUR, EUR/MWh]; the stage after T is stage 1, and ``end_value_keur`` ``(K,)`` is the value
    after stage T that the last cycle used. ``next_index``, ``release_gwh`` and ``spill_gwh`` are
    the optimal decisions ``(T, K, J)``. ``history_eur_mwh`` holds the largest change of any
    water value between consecutive cycles.

    ``value_keur`` is the expected cost-to-go only up to an additive constant: the cycles stop
    when the water values settle, while the level of V still drifts (it contracts by only
    ``delta^T`` per cycle). Only differences of V, such as the water values or the comparison
    of decisions within a stage, are meaningful.
    """

    problem: SdpProblem
    value_keur: np.ndarray
    end_value_keur: np.ndarray
    water_value_eur_mwh: np.ndarray
    next_index: np.ndarray
    release_gwh: np.ndarray
    spill_gwh: np.ndarray
    history_eur_mwh: tuple[float, ...]
    cycles: int
    converged: bool

    @property
    def next_storage_gwh(self) -> np.ndarray:
        """Optimal storage at the end of each stage ``(T, K, J)`` [GWh]."""
        return self.problem.storage_gwh[self.next_index]

    def value_after(self, t: int) -> np.ndarray:
        """Value on the grid after stage ``t`` (0-based) [kEUR, up to a constant]."""
        if t == self.problem.stages - 1:
            return self.end_value_keur
        return self.value_keur[t + 1]


def solve(problem: SdpProblem, tol_eur_mwh: float, max_cycles: int) -> SdpSolution:
    """Solve the periodic year: repeat the backward pass until the water values settle.

    The first cycle ends with a zero value; each later one ends with the value of stage 1 of
    the cycle before.

    Parameters
    ----------
    problem : SdpProblem
        The inputs.
    tol_eur_mwh : float
        Stop when no water value changes more than this between cycles [EUR/MWh].
    max_cycles : int
        Upper bound on the number of cycles.

    Returns
    -------
    SdpSolution
        ``converged`` is False if ``max_cycles`` was reached first.
    """
    end_value = np.zeros(len(problem.storage_gwh))
    previous: np.ndarray | None = None
    history: list[float] = []
    converged = False
    cycles = 0
    for _ in range(max_cycles):
        cycles += 1
        used_end_value = end_value
        result = backward_pass(problem, used_end_value)
        w = water_values(result.value_keur[:-1], problem.storage_gwh)
        if previous is not None:
            history.append(float(np.abs(w - previous).max()))
            if history[-1] < tol_eur_mwh:
                converged = True
                break
        previous = w
        end_value = result.value_keur[0]
    return SdpSolution(
        problem=problem,
        value_keur=result.value_keur[:-1],
        end_value_keur=used_end_value,
        water_value_eur_mwh=w,
        next_index=result.next_index,
        release_gwh=result.release_gwh,
        spill_gwh=result.spill_gwh,
        history_eur_mwh=tuple(history),
        cycles=cycles,
        converged=converged,
    )


def first_order_check(solution: SdpSolution) -> dict[str, np.ndarray]:
    """Check the optimality condition ``p_t = delta w_{t+1}(S')`` on the grid.

    Moving the next storage one grid step ``dS`` down (releasing ``dS`` more) or up must not
    lower the cost. With the immediate cost quadratic in the release, this brackets the week's
    price between the discounted one-sided water values of next week:

        delta w_right(S') - b_t dS / 2  <=  p_t  <=  delta w_left(S') + b_t dS / 2,

    with ``w_left = -(V(S') - V(S' - dS)) / dS`` and ``w_right = -(V(S' + dS) - V(S')) / dS``.

    Parameters
    ----------
    solution : SdpSolution
        A solved problem.

    Returns
    -------
    dict of str to numpy.ndarray
        ``(T, K, J)`` arrays: ``price`` [EUR/MWh] at the optimal release, ``lower`` and
        ``upper`` (the bracket) [EUR/MWh], ``centred`` (the discounted centred water value)
        [EUR/MWh], and ``interior``: decisions where both neighbouring moves keep the release
        within its bounds, without spill, and the next storage off the grid ends.
    """
    p = solution.problem
    s = p.storage_gwh
    step = s[1] - s[0]
    release, spill, index = solution.release_gwh, solution.spill_gwh, solution.next_index
    stages = np.arange(p.stages)[:, None, None]
    slope = p.slope_eur_mwh_per_gwh[stages]
    price = p.intercept_eur_mwh[stages] + slope * (p.residual_demand_gwh[stages] - release)
    value_next = np.vstack([solution.value_keur[1:], solution.end_value_keur[None, :]])
    flat = np.clip(index, 1, len(s) - 2).reshape(p.stages, -1)

    def value_at(offset: int) -> np.ndarray:
        return np.take_along_axis(value_next, flat + offset, axis=1).reshape(index.shape)

    w_left = -(value_at(0) - value_at(-1)) / step
    w_right = -(value_at(1) - value_at(0)) / step
    available = s[None, :, None] + p.inflow_gwh[:, None, :]
    lower_release = np.minimum(p.release_min_gwh[stages], available)
    interior = (
        (release - step >= lower_release)
        & (release + step <= p.release_max_gwh[stages])
        & (spill == 0)
        & (index > 0)
        & (index < len(s) - 1)
    )
    return {
        "price": price,
        "lower": p.discount * w_right - slope * step / 2,
        "upper": p.discount * w_left + slope * step / 2,
        "centred": p.discount * (w_left + w_right) / 2,
        "interior": interior,
    }


def grid_resolution_eur_mwh(problem: SdpProblem) -> float:
    """Price resolution of water values on the grid: half a storage step of the flattest curve.

    The policy is restricted to grid points, so a water value is only resolved to about the
    price change that half a storage step of release causes [EUR/MWh].
    """
    return float(problem.slope_eur_mwh_per_gwh.min() * (problem.storage_gwh[1] / 2))


def checks(solution: SdpSolution) -> dict[str, float]:
    """Sanity checks of a solution: monotone water values and the optimality condition.

    Returns
    -------
    dict of str to float
        ``resolution_eur_mwh`` (:func:`grid_resolution_eur_mwh`); ``rising_cells`` and
        ``max_rise_eur_mwh``: water values that grow with storage by more than the resolution,
        and the largest growth; ``interior_decisions``, ``within_bracket_share`` and
        ``median_gap_eur_mwh`` from :func:`first_order_check` (median of
        ``|p_t - delta w_{t+1}(S')|`` with centred water values); ``max_rise_week`` and
        ``max_rise_filling_pct``: where the largest rise is; ``min_water_value_eur_mwh``: with
        spill as an option, no MWh is worth less than ``-c_sp``.
    """
    resolution = grid_resolution_eur_mwh(solution.problem)
    rise = np.diff(solution.water_value_eur_mwh, axis=1)
    week, level = np.unravel_index(int(rise.argmax()), rise.shape)
    s = solution.problem.storage_gwh
    check = first_order_check(solution)
    interior = check["interior"]
    price = check["price"][interior]
    within = (price >= check["lower"][interior]) & (price <= check["upper"][interior])
    return {
        "resolution_eur_mwh": resolution,
        "rising_cells": float((rise > resolution).sum()),
        "max_rise_eur_mwh": float(rise.max()),
        "interior_decisions": float(interior.sum()),
        "within_bracket_share": float(within.mean()) if interior.any() else float("nan"),
        "median_gap_eur_mwh": float(np.median(np.abs(price - check["centred"][interior]))),
        "max_rise_week": float(week + 1),
        "max_rise_filling_pct": float(100 * s[level + 1] / s[-1]),
        "min_water_value_eur_mwh": float(solution.water_value_eur_mwh.min()),
    }


def grid_sensitivity(solutions: Mapping[int, SdpSolution], reference: int) -> pd.DataFrame:
    """Difference of water values between storage grids, on the reference grid's fillings.

    Parameters
    ----------
    solutions : mapping of int to SdpSolution
        Solutions of the same problem with different numbers of storage points.
    reference : int
        Key of the solution compared against (normally the finest grid).

    Returns
    -------
    pandas.DataFrame
        Indexed by the number of points: ``max_abs_diff_eur_mwh`` and ``mean_abs_diff_eur_mwh``
        over all stages and storage levels [EUR/MWh].
    """
    ref = solutions[reference]
    filling = ref.problem.storage_gwh / ref.problem.capacity_gwh
    rows = {}
    for points, solution in sorted(solutions.items()):
        own = solution.problem.storage_gwh / solution.problem.capacity_gwh
        w = np.array([np.interp(filling, own, row) for row in solution.water_value_eur_mwh])
        diff = np.abs(w - ref.water_value_eur_mwh)
        rows[points] = {
            "max_abs_diff_eur_mwh": float(diff.max()),
            "mean_abs_diff_eur_mwh": float(diff.mean()),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("n_storage_points")


def compare_solutions(
    solutions: Mapping[str, SdpSolution], points: Sequence[tuple[int, float]]
) -> pd.DataFrame:
    """Tabulate the checks and selected water values of several solutions, side by side.

    Parameters
    ----------
    solutions : mapping of str to SdpSolution
        Label to solution.
    points : sequence of (int, float)
        ``(ISO week, filling [%])`` pairs to tabulate.

    Returns
    -------
    pandas.DataFrame
        One row per label: ``rising_cells`` and ``max_rise_eur_mwh`` (see :func:`checks`) and
        the water value at each point [EUR/MWh].
    """
    rows = {}
    for label, solution in solutions.items():
        found = checks(solution)
        rows[label] = {
            "rising_cells": int(found["rising_cells"]),
            "max_rise_eur_mwh": found["max_rise_eur_mwh"],
            **{
                f"week {week} at {pct:g}%": water_value_at(solution, week, pct)
                for week, pct in points
            },
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _span(years: Sequence[int]) -> str:
    """``2021-2023`` for a run of years."""
    ordered = sorted(years)
    return f"{ordered[0]}-{ordered[-1]}" if ordered else "no years"


def capacity_gwh(cfg: AppConfig, weekly: pd.DataFrame) -> float:
    """``S_max``: ``reservoir.capacity_gwh``, or the largest storage observed when it is null."""
    if cfg.reservoir.capacity_gwh is not None:
        return float(cfg.reservoir.capacity_gwh)
    return float(max(weekly["storage_gwh"].max(), weekly["storage_end_gwh"].max()))


def release_min(
    hydro_stage_gwh: pd.Series,
    inflow_stage_gwh: pd.Series,
    iso_week: pd.Series,
    percentile: float,
    basis: str,
) -> np.ndarray:
    """Minimum release of each ISO week, ``R_min_t`` [GWh per 168-hour stage].

    Parameters
    ----------
    hydro_stage_gwh, inflow_stage_gwh : pandas.Series
        Weekly hydro production and inflow per stage [GWh]; weeks without production are
        skipped.
    iso_week : pandas.Series
        ISO week (1-52) of each row.
    percentile : float
        Percentile over the years [0-100].
    basis : {"production", "inflow_capped"}
        ``production``: the percentile of production (the guide's definition).
        ``inflow_capped``: the percentile of ``min(production, inflow)``, since forced
        production cannot exceed what flows in during the week.

    Returns
    -------
    numpy.ndarray
        Shape ``(52,)``.
    """
    if basis == "production":
        forced = hydro_stage_gwh
    elif basis == "inflow_capped":
        forced = hydro_stage_gwh.where(hydro_stage_gwh <= inflow_stage_gwh, inflow_stage_gwh)
        forced = forced.where(hydro_stage_gwh.notna())
    else:
        raise SdpError(f"unknown minimum-release basis: {basis!r}")
    weeks = range(1, MODEL_WEEKS_PER_YEAR + 1)
    values = forced.groupby(iso_week).quantile(percentile / 100).reindex(weeks)
    if values.isna().any():
        raise SdpError(f"no production data for weeks {values.index[values.isna()].tolist()}")
    return values.to_numpy(dtype=float)


def base_de_price_profile(
    cfg: AppConfig, table: pd.DataFrame, level_eur_mwh: float | None = None
) -> np.ndarray:
    """German price per stage for the water values [EUR/MWh].

    The shape is the mean per ISO week over the years of ``sdp.de_price_shape`` (all years, or
    those outside ``price_model.crisis_years``); the level is ``level_eur_mwh``, else
    ``sdp.de_price_level_eur_mwh``, else the mean of the all-years profile.

    Parameters
    ----------
    cfg : AppConfig
        Configuration.
    table : pandas.DataFrame
        Output of :func:`watervalues.price.price_table`.
    level_eur_mwh : float or None
        Annual mean of the profile [EUR/MWh] (the continental price scenarios).

    Returns
    -------
    numpy.ndarray
        Shape ``(52,)``, position = ISO week - 1.
    """
    level = level_eur_mwh or cfg.sdp.de_price_level_eur_mwh
    if level is None:
        level = float(de_price_profile(table).mean())
    years = sorted(int(year) for year in table["iso_year"].unique())
    if cfg.sdp.de_price_shape == "without_crisis":
        years = [year for year in years if year not in cfg.price_model.crisis_years]
    return de_price_profile(table, years=years, level_eur_mwh=level)


def de_price_levels(cfg: AppConfig, table: pd.DataFrame) -> dict[str, float]:
    """Mean German price of all years and of the years outside the crisis [EUR/MWh].

    Parameters
    ----------
    cfg : AppConfig
        Configuration (``price_model.crisis_years``).
    table : pandas.DataFrame
        Output of :func:`watervalues.price.price_table`.

    Returns
    -------
    dict of str to float
        ``all_years`` and ``without_crisis``: means of the weekly profiles over 52 stages.
    """
    years = sorted(int(year) for year in table["iso_year"].unique())
    calm = [year for year in years if year not in cfg.price_model.crisis_years]
    return {
        "all_years": float(de_price_profile(table).mean()),
        "without_crisis": float(de_price_profile(table, years=calm).mean()),
    }


def build_problem(
    cfg: AppConfig,
    weekly: pd.DataFrame,
    *,
    n_storage_points: int | None = None,
    de_price_eur_mwh: np.ndarray | None = None,
    curve: PriceCurve | None = None,
) -> SdpProblem:
    """Assemble the SDP inputs from the weekly dataset and the configuration.

    - inflow scenarios: the matrix of :func:`watervalues.inflow.build_inflow` (one equally
      likely scenario per ISO year);
    - ``D_t``: mean residual demand of each ISO week over the years with data;
    - ``R_min_t``: :func:`release_min` with ``reservoir.rmin_percentile`` and ``rmin_basis``;
    - ``R_max``: installed capacity x 168 h x availability;
    - price curve: the configured base model (:mod:`watervalues.price`) with the German price
      profile of :func:`base_de_price_profile`.

    Parameters
    ----------
    cfg : AppConfig
        Configuration.
    weekly : pandas.DataFrame
        Weekly dataset (``weekly_se.csv`` contract) covering ``cfg.period``.
    n_storage_points : int or None
        Grid size; ``sdp.n_storage_points`` when None.
    de_price_eur_mwh : numpy.ndarray or None
        German price per stage ``(52,)`` [EUR/MWh]; the configured profile when None.
    curve : PriceCurve or None
        Price curve; the base model fitted on all of ``weekly`` when None.

    Returns
    -------
    SdpProblem
    """
    pm, res, sdp = cfg.price_model, cfg.reservoir, cfg.sdp
    table = price_table(weekly, pm.seasons)
    if curve is None:
        curve = fit_curve(table, base_spec(cfg), pm.hac_maxlags)
    if de_price_eur_mwh is None:
        de_price_eur_mwh = base_de_price_profile(cfg, table)
    stage = curve.stage_curve(de_price_eur_mwh)
    inflow = build_inflow(cfg, weekly)
    scenarios = inflow.scenarios  # (J, 52)
    weeks = range(1, MODEL_WEEKS_PER_YEAR + 1)
    demand = table.groupby("iso_week")["residual_demand_stage_gwh"].mean().reindex(weeks)
    minimum = release_min(
        table["hydro_stage_gwh"],
        inflow.weekly["inflow_stage_gwh"],
        table["iso_week"],
        res.rmin_percentile,
        res.rmin_basis,
    )
    release_max = res.hydro_capacity_mw * HOURS_PER_WEEK * res.availability / MWH_PER_GWH
    points = n_storage_points or sdp.n_storage_points
    return SdpProblem(
        storage_gwh=np.linspace(0.0, capacity_gwh(cfg, weekly), points),
        inflow_gwh=scenarios.to_numpy().T,
        probability=np.full(len(scenarios), 1.0 / len(scenarios)),
        residual_demand_gwh=demand.to_numpy(),
        release_min_gwh=minimum,
        release_max_gwh=np.full(MODEL_WEEKS_PER_YEAR, release_max),
        intercept_eur_mwh=stage.intercept_eur_mwh,
        slope_eur_mwh_per_gwh=stage.slope_eur_mwh_per_gwh,
        spill_penalty_eur_mwh=sdp.spill_penalty_eur_mwh,
        discount=(1 + sdp.discount_rate_annual) ** (-1 / MODEL_WEEKS_PER_YEAR),
    )


def timed_solve(cfg: AppConfig, problem: SdpProblem) -> tuple[SdpSolution, float]:
    """Solve with the configured tolerance and cycle limit; also return the time taken [s]."""
    start = time.perf_counter()
    solution = solve(problem, cfg.sdp.tol_eur_mwh, cfg.sdp.max_cycles)
    return solution, time.perf_counter() - start


def water_value_table(solution: SdpSolution) -> pd.DataFrame:
    """Water values in long form: one row per ISO week and storage point.

    Returns
    -------
    pandas.DataFrame
        ``iso_week``, ``storage_gwh`` [GWh], ``filling_pct`` [% of ``S_max``] and
        ``water_value_eur_mwh`` [EUR/MWh]. The level of V is left out: it is not converged
        (see :class:`SdpSolution`).
    """
    s = solution.problem.storage_gwh
    weeks, points = solution.water_value_eur_mwh.shape
    return pd.DataFrame(
        {
            "iso_week": np.repeat(np.arange(1, weeks + 1), points),
            "storage_gwh": np.tile(s, weeks),
            "filling_pct": np.tile(100 * s / s[-1], weeks),
            "water_value_eur_mwh": solution.water_value_eur_mwh.ravel(),
        }
    )


def water_value_at(solution: SdpSolution, week: int, filling_pct: float) -> float:
    """Water value of an ISO week at a storage level, interpolated linearly [EUR/MWh]."""
    s = solution.problem.storage_gwh
    return float(np.interp(filling_pct / 100 * s[-1], s, solution.water_value_eur_mwh[week - 1]))


def sdp_report(
    cfg: AppConfig, weekly: pd.DataFrame, solution: SdpSolution, seconds: float
) -> list[str]:
    """Human-readable summary: inputs, convergence and water values at chosen levels.

    Parameters
    ----------
    cfg : AppConfig
        Configuration (``sdp``, ``reservoir``, ``data.sanity``).
    weekly : pandas.DataFrame
        The weekly dataset (to say where ``S_max`` comes from).
    solution : SdpSolution
        The solved base case.
    seconds : float
        Solve time [s].

    Returns
    -------
    list of str
    """
    p = solution.problem
    s = p.storage_gwh
    source = (
        "reservoir.capacity_gwh"
        if cfg.reservoir.capacity_gwh is not None
        else "largest storage observed"
    )
    reference = cfg.data.sanity.storage_capacity_ref_gwh
    years = cfg.period.first_iso_year, cfg.period.last_iso_year
    table = price_table(weekly, cfg.price_model.seasons)
    levels = de_price_levels(cfg, table)
    full = p.intercept_eur_mwh + p.slope_eur_mwh_per_gwh * (
        p.residual_demand_gwh - p.release_max_gwh
    )
    lines = [
        "Water value SDP: V_t(S) = sum_j pi_j min [C_t(D_t - R) + c_sp Sp + delta V_t+1(S')]; "
        "w = -dV/dS [EUR/MWh]",
        f"  storage grid: {len(s)} points, 0 to {p.capacity_gwh:,.0f} GWh ({source}; reference "
        f"capacity {reference:,.0f} GWh), step {s[1]:.1f} GWh",
        f"  inflow scenarios: {p.inflow_gwh.shape[1]} ISO years ({years[0]}-{years[1]}), "
        "probability 1/J each, independent between weeks",
        f"  discount per stage {p.discount:.6f} ({cfg.sdp.discount_rate_annual:.0%} per year); "
        f"spill penalty {p.spill_penalty_eur_mwh:g} EUR/MWh",
        f"  release: R_max {p.release_max_gwh.max():,.0f} GWh per stage; R_min "
        f"{p.release_min_gwh.min():,.0f} to {p.release_min_gwh.max():,.0f} GWh "
        f"(percentile {cfg.reservoir.rmin_percentile:g}, basis {cfg.reservoir.rmin_basis})",
        f"  residual demand D_t: {p.residual_demand_gwh.min():,.0f} to "
        f"{p.residual_demand_gwh.max():,.0f} GWh per stage",
        f"  German price profile (configured): shape {cfg.sdp.de_price_shape}, mean "
        f"{base_de_price_profile(cfg, table).mean():.1f} EUR/MWh; the mean of all years, "
        f"{levels['all_years']:.1f}, includes {_span(cfg.price_model.crisis_years)}; "
        f"the other years average {levels['without_crisis']:.1f}",
        f"  price curve: a_t {p.intercept_eur_mwh.min():.1f} to {p.intercept_eur_mwh.max():.1f} "
        f"EUR/MWh, b_t {p.slope_eur_mwh_per_gwh.min():.4f} to "
        f"{p.slope_eur_mwh_per_gwh.max():.4f} EUR/MWh per GWh",
        f"  price at full release p_t(D_t - R_max): {full.min():.1f} to {full.max():.1f} EUR/MWh, "
        f"below zero in {int((full < 0).sum())} of {p.stages} weeks (the linear curve beyond "
        f"the largest observed export, {table['net_import_stage_gwh'].min():,.0f} GWh); there "
        "the model spills rather than sells below -c_sp",
        f"\nConvergence: {solution.cycles} cycles, "
        f"{'converged' if solution.converged else 'NOT converged'} (tolerance "
        f"{cfg.sdp.tol_eur_mwh:g} EUR/MWh); solved in {seconds:.1f} s",
        "  largest water value change per cycle [EUR/MWh]: "
        + ", ".join(f"{change:.3g}" for change in solution.history_eur_mwh),
    ]
    fillings = cfg.sdp.report_fillings_pct
    table = pd.DataFrame(
        {
            f"{pct:g}%": [water_value_at(solution, week, pct) for week in cfg.sdp.curve_weeks]
            for pct in fillings
        },
        index=pd.Index(cfg.sdp.curve_weeks, name="iso_week"),
    )
    found = checks(solution)
    floor_ok = found["min_water_value_eur_mwh"] >= -p.spill_penalty_eur_mwh - _VALUE_TOL
    lines += [
        "\nChecks: water values non-increasing in storage within the grid resolution "
        f"({found['resolution_eur_mwh']:.2f} EUR/MWh): "
        f"{'yes' if found['rising_cells'] == 0 else 'NO'} ({found['rising_cells']:.0f} cells "
        f"above; largest rise {found['max_rise_eur_mwh']:.2f} EUR/MWh, in week "
        f"{found['max_rise_week']:.0f} at {found['max_rise_filling_pct']:.1f}% filling)",
        f"  water values >= -c_sp (spill is always possible): {'yes' if floor_ok else 'NO'} "
        f"(lowest {found['min_water_value_eur_mwh']:.3f} EUR/MWh)",
        f"  optimality p_t = delta w_t+1(S') at {found['interior_decisions']:.0f} interior "
        f"decisions: {found['within_bracket_share']:.2%} inside the grid bracket; median "
        f"|p_t - delta w_t+1| {found['median_gap_eur_mwh']:.2f} EUR/MWh",
    ]
    w = solution.water_value_eur_mwh
    lines += [
        "\nWater values [EUR/MWh] by ISO week and storage level [% of S_max]",
        table.round(1).to_string(),
        f"  all stages and levels: {w.min():.1f} to {w.max():.1f} EUR/MWh; median "
        f"{np.median(w):.1f}",
    ]
    return lines


def save_solution(solution: SdpSolution, path: Path) -> None:
    """Write a solution and its problem as a compressed NumPy archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    p = solution.problem
    arrays = {f"problem_{f.name}": np.asarray(getattr(p, f.name)) for f in fields(p)}
    np.savez_compressed(
        path,
        **arrays,
        value_keur=solution.value_keur,
        end_value_keur=solution.end_value_keur,
        water_value_eur_mwh=solution.water_value_eur_mwh,
        next_index=solution.next_index,
        release_gwh=solution.release_gwh,
        spill_gwh=solution.spill_gwh,
        history_eur_mwh=np.asarray(solution.history_eur_mwh),
        cycles=np.asarray(solution.cycles),
        converged=np.asarray(solution.converged),
    )


def load_solution(path: Path) -> SdpSolution:
    """Read a solution written by :func:`save_solution`."""
    with np.load(path) as data:
        problem = SdpProblem(
            **{
                f.name: (
                    data[f"problem_{f.name}"]
                    if f.name in _ARRAYS
                    else float(data[f"problem_{f.name}"])
                )
                for f in fields(SdpProblem)
            }
        )
        return SdpSolution(
            problem=problem,
            value_keur=data["value_keur"],
            end_value_keur=data["end_value_keur"],
            water_value_eur_mwh=data["water_value_eur_mwh"],
            next_index=data["next_index"],
            release_gwh=data["release_gwh"],
            spill_gwh=data["spill_gwh"],
            history_eur_mwh=tuple(float(x) for x in data["history_eur_mwh"]),
            cycles=int(data["cycles"]),
            converged=bool(data["converged"]),
        )


def write_outputs(cfg: AppConfig, solution: SdpSolution, report: list[str]) -> None:
    """Write the archive, the water value table and the report (paths in ``cfg.sdp``)."""
    sdp = cfg.sdp
    save_solution(solution, PROJECT_ROOT / sdp.values_npz)
    table_path = PROJECT_ROOT / sdp.water_values_csv
    table_path.parent.mkdir(parents=True, exist_ok=True)
    water_value_table(solution).to_csv(
        table_path, index=False, float_format=cfg.data.csv_float_format, lineterminator="\n"
    )
    report_path = PROJECT_ROOT / sdp.report
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8", newline="\n")
