"""Reduced-form price curve of the Swedish market and the cost of covering net import.

Every variant of the curve (guide section 5.2, extended with the owner-approved proportional
slope) is

    p_w = alpha + s(w) + gamma p_DE,w + b_w N_w + e_w,

with

- ``N_w`` the net import per 168-hour stage [GWh]: residual demand (load - nuclear - wind - solar
  - other) minus hydro production, negative when Sweden exports;
- ``p_w`` the load-weighted Swedish price and ``p_DE,w`` the German price [EUR/MWh];
- ``s(w)`` the intercept of the season of ISO week ``w`` [EUR/MWh] (optional dummies; the first
  season is the reference);
- ``b_w`` the slope: constant, ``b = beta`` [EUR/MWh per GWh] (the guide's form), or
  proportional to the German price, ``b_w = kappa p_DE,w`` with ``kappa`` [1/GWh].

For a 52-stage German price profile every variant reduces to a weekly intercept and slope,
``a_t = alpha + s(t) + gamma p_DE,t`` and ``b_t`` (:class:`StageCurve`), and to the cost of
covering ``N`` in stage ``t``,

    C_t(N) = a_t N + b_t / 2 N^2  [kEUR],    C_t'(N) = p_t(N) = a_t + b_t N,

which is convex when ``b_t > 0``. With ``N < 0`` the cost is negative: it is export revenue.

Weeks with a missing input are left out of the regression and listed; nothing is imputed.
Coefficients are OLS estimates. Standard errors are HAC (Newey-West, Bartlett kernel), because
the weekly residuals are autocorrelated. Out-of-sample accuracy is measured by leaving one ISO
year out at a time (LOYO).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.typing import ArrayLike
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.stattools import durbin_watson

from watervalues.config import PROJECT_ROOT, AppConfig
from watervalues.data.aggregate import per_stage
from watervalues.data.build import derived_columns, week_label, write_weekly_csv
from watervalues.units import EUR_PER_KEUR, MODEL_WEEKS_PER_YEAR, MWH_PER_GWH

SlopeForm = Literal["constant", "proportional"]
SLOPE_FORMS: tuple[SlopeForm, ...] = ("constant", "proportional")
# Every design column is named after its coefficient.
ALPHA, GAMMA, BETA, KAPPA = "alpha", "gamma", "beta", "kappa"
SLOPE_TERM: dict[str, str] = {"constant": BETA, "proportional": KAPPA}
SEASON_PREFIX = "season_"
UNITS = {
    ALPHA: "EUR/MWh",
    SEASON_PREFIX: "EUR/MWh",
    GAMMA: "-",
    BETA: "EUR/MWh per GWh",
    KAPPA: "1/GWh",
}
# Columns of the weekly dataset behind N, p and p_DE: a week missing any of them is left out.
REGRESSION_INPUTS = (
    "hydro_gwh",
    "load_gwh",
    "nuclear_gwh",
    "wind_gwh",
    "solar_gwh",
    "other_gwh",
    "price_se_eur_mwh",
    "price_de_eur_mwh",
)
# Status of a week in the base model's sample.
USED, MISSING, EXCLUDED = "used", "missing", "excluded"


class PriceError(RuntimeError):
    """A check on the price curve failed; the curve must not be used."""


def _year_span(years: Sequence[int]) -> str:
    """``2021-2023`` for consecutive years, otherwise ``2019, 2021``."""
    ordered = sorted(years)
    if len(ordered) > 1 and ordered == list(range(ordered[0], ordered[-1] + 1)):
        return f"{ordered[0]}-{ordered[-1]}"
    return ", ".join(str(year) for year in ordered)


def _weeks(mask: np.ndarray) -> list[int]:
    """ISO weeks (1-52) where a 52-stage mask is true."""
    return [int(i) + 1 for i in np.flatnonzero(mask)]


def _stage_index(week: ArrayLike) -> np.ndarray:
    """Zero-based stage index of one or more ISO weeks 1-52."""
    weeks = np.asarray(week)
    if weeks.dtype.kind not in "iu":
        raise ValueError(f"week must be an integer ISO week, got {weeks.dtype}")
    if ((weeks < 1) | (weeks > MODEL_WEEKS_PER_YEAR)).any():
        raise ValueError(f"week must be in 1-{MODEL_WEEKS_PER_YEAR}: {np.unique(weeks).tolist()}")
    return weeks - 1


def season_of_week(iso_week: ArrayLike, seasons: Sequence[tuple[str, int]]) -> np.ndarray:
    """Season name of each ISO week.

    Parameters
    ----------
    iso_week : array_like of int
        ISO weeks 1-52.
    seasons : sequence of (str, int)
        Season names and their first ISO weeks, the first starting at week 1.

    Returns
    -------
    numpy.ndarray
        The season name of each week.
    """
    names = np.array([name for name, _ in seasons])
    first_weeks = np.array([first for _, first in seasons])
    return names[np.searchsorted(first_weeks, _stage_index(iso_week) + 1, side="right") - 1]


@dataclass(frozen=True)
class CurveSpec:
    """One variant of the price curve.

    Attributes
    ----------
    slope_form : {"constant", "proportional"}
        ``constant``: slope ``beta`` [EUR/MWh per GWh]; ``proportional``: slope ``kappa * p_DE``
        with ``kappa`` [1/GWh].
    seasons : tuple of (str, int)
        Season names and first ISO weeks for the seasonal dummies; empty for none.
    exclude_years : tuple of int
        ISO years left out of the fit.
    """

    slope_form: SlopeForm
    seasons: tuple[tuple[str, int], ...] = ()
    exclude_years: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.slope_form not in SLOPE_FORMS:
            raise ValueError(f"slope_form must be one of {SLOPE_FORMS}: {self.slope_form!r}")
        seasons = tuple((str(name), int(first)) for name, first in self.seasons)
        firsts = [first for _, first in seasons]
        if seasons and (firsts[0] != 1 or firsts != sorted(set(firsts))):
            raise ValueError(f"seasons must start at ISO week 1 and increase: {seasons}")
        object.__setattr__(self, "seasons", seasons)
        object.__setattr__(
            self, "exclude_years", tuple(sorted({int(y) for y in self.exclude_years}))
        )

    @property
    def slope_term(self) -> str:
        """Name of the slope coefficient: ``beta`` or ``kappa``."""
        return SLOPE_TERM[self.slope_form]

    @property
    def terms(self) -> tuple[str, ...]:
        """Coefficient names, in the column order of :func:`design_matrix`."""
        seasons = tuple(SEASON_PREFIX + name for name, _ in self.seasons[1:])
        return (ALPHA, *seasons, GAMMA, self.slope_term)

    @property
    def label(self) -> str:
        """Short name, e.g. ``proportional slope, seasonal, all years``."""
        seasons = "seasonal" if self.seasons else "no seasons"
        sample = f"without {_year_span(self.exclude_years)}" if self.exclude_years else "all years"
        return f"{self.slope_form} slope, {seasons}, {sample}"

    def to_dict(self) -> dict:
        """JSON-ready form."""
        return {
            "slope_form": self.slope_form,
            "seasons": dict(self.seasons),
            "exclude_years": list(self.exclude_years),
        }

    @classmethod
    def from_dict(cls, raw: Mapping) -> CurveSpec:
        """Inverse of :meth:`to_dict`."""
        return cls(raw["slope_form"], tuple(raw["seasons"].items()), tuple(raw["exclude_years"]))


def without_year(spec: CurveSpec, year: int) -> CurveSpec:
    """Return the same variant with one more ISO year left out (leave-one-year-out refits)."""
    return replace(spec, exclude_years=(*spec.exclude_years, int(year)))


def price_table(weekly: pd.DataFrame, seasons: Mapping[str, int]) -> pd.DataFrame:
    """Regression table: net import per 168-hour stage, the two prices and the season.

    Residual demand and net import come from :func:`~watervalues.data.build.derived_columns`
    on the unscaled rows; :func:`~watervalues.data.aggregate.per_stage` then rescales them, so a
    folded week 52 (336 h) contributes the mean of its two weeks. Its prices are 336-hour means
    and stay as they are.

    Parameters
    ----------
    weekly : pandas.DataFrame
        Weekly dataset (``weekly_se.csv`` contract): energies [GWh over the row's hours], prices
        [EUR/MWh].
    seasons : mapping of str to int
        Season name to its first ISO week (``price_model.seasons``).

    Returns
    -------
    pandas.DataFrame
        ``week_start``, ``iso_year``, ``iso_week`` and ``hours`` [h]; ``residual_demand_stage_gwh``,
        ``hydro_stage_gwh`` and ``net_import_stage_gwh`` [GWh per 168-hour stage];
        ``price_se_eur_mwh`` and ``price_de_eur_mwh`` [EUR/MWh]; ``season``; ``missing_inputs``,
        the missing columns of :data:`REGRESSION_INPUTS` ("" if none), and ``complete``.
    """
    flows = per_stage(
        derived_columns(weekly), ["residual_demand_gwh", "hydro_gwh", "net_import_gwh"]
    )
    out = weekly[["week_start", "iso_year", "iso_week", "hours"]].copy()
    for name in ("residual_demand", "hydro", "net_import"):
        out[f"{name}_stage_gwh"] = flows[f"{name}_gwh"]
    out["price_se_eur_mwh"] = weekly["price_se_eur_mwh"]
    out["price_de_eur_mwh"] = weekly["price_de_eur_mwh"]
    out["season"] = season_of_week(weekly["iso_week"].to_numpy(), tuple(seasons.items()))
    missing = weekly[list(REGRESSION_INPUTS)].isna()
    out["missing_inputs"] = missing.apply(lambda row: ", ".join(row.index[row]), axis=1)
    out["complete"] = ~missing.any(axis=1)
    return out


def design_matrix(table: pd.DataFrame, spec: CurveSpec) -> pd.DataFrame:
    """Regressors of a variant, each column named after its coefficient.

    Parameters
    ----------
    table : pandas.DataFrame
        With ``iso_week``, ``net_import_stage_gwh`` [GWh per stage] and ``price_de_eur_mwh``
        [EUR/MWh] (see :func:`price_table`).
    spec : CurveSpec
        The variant.

    Returns
    -------
    pandas.DataFrame
        Columns :attr:`CurveSpec.terms`: ``alpha`` (ones), ``season_<name>`` (0/1, every season
        after the reference), ``gamma`` (German price [EUR/MWh]) and the slope regressor, ``beta``
        (net import [GWh]) or ``kappa`` (net import x German price [GWh EUR/MWh]).
    """
    columns: dict[str, np.ndarray] = {ALPHA: np.ones(len(table))}
    if spec.seasons:
        season = season_of_week(table["iso_week"].to_numpy(), spec.seasons)
        for name, _ in spec.seasons[1:]:
            columns[SEASON_PREFIX + name] = (season == name).astype(float)
    price_de = table["price_de_eur_mwh"].to_numpy(dtype=float)
    net_import = table["net_import_stage_gwh"].to_numpy(dtype=float)
    columns[GAMMA] = price_de
    columns[spec.slope_term] = (
        net_import if spec.slope_form == "constant" else net_import * price_de
    )
    return pd.DataFrame(columns, index=table.index)


def _sample(table: pd.DataFrame, spec: CurveSpec) -> np.ndarray:
    """Rows a variant is fitted on: complete weeks outside its excluded years."""
    absent = sorted(set(spec.exclude_years) - set(table["iso_year"]))
    if absent:
        raise PriceError(f"{spec.label}: the excluded years {absent} are not in the data")
    return (table["complete"] & ~table["iso_year"].isin(spec.exclude_years)).to_numpy()


def _ols(
    table: pd.DataFrame, spec: CurveSpec, hac_maxlags: int
) -> tuple[RegressionResultsWrapper, np.ndarray]:
    """OLS fit with HAC standard errors, and the mask of the rows used."""
    used = _sample(table, spec)
    x = design_matrix(table[used], spec)
    if len(x) <= x.shape[1] or np.linalg.matrix_rank(x.to_numpy()) < x.shape[1]:
        raise PriceError(
            f"{spec.label}: {len(x)} weeks do not identify the {x.shape[1]} coefficients "
            "(rank-deficient design)"
        )
    y = table.loc[used, "price_se_eur_mwh"]
    results = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": hac_maxlags})
    return results, used


@dataclass(frozen=True)
class StageCurve:
    """Weekly intercept and slope of the price curve for one German price profile.

    Every array has one value per stage (shape ``(52,)``, position = ISO week - 1). They are
    copied and read-only.

    Attributes
    ----------
    intercept_eur_mwh : numpy.ndarray
        ``a_t = alpha + s(t) + gamma p_DE,t``: the price at zero net import [EUR/MWh].
    slope_eur_mwh_per_gwh : numpy.ndarray
        ``b_t`` [EUR/MWh per GWh of net import per stage]; must be > 0 in every stage.
    de_price_eur_mwh : numpy.ndarray
        The German price profile [EUR/MWh].

    Raises
    ------
    PriceError
        If an array does not have 52 finite values, or a slope is <= 0 (non-convex cost).
    """

    intercept_eur_mwh: np.ndarray
    slope_eur_mwh_per_gwh: np.ndarray
    de_price_eur_mwh: np.ndarray

    def __post_init__(self) -> None:
        for name in ("intercept_eur_mwh", "slope_eur_mwh_per_gwh", "de_price_eur_mwh"):
            values = np.array(getattr(self, name), dtype=float)
            if values.shape != (MODEL_WEEKS_PER_YEAR,):
                raise PriceError(
                    f"{name} needs one value per stage ({MODEL_WEEKS_PER_YEAR}), got shape "
                    f"{values.shape}"
                )
            finite = np.isfinite(values)
            if not finite.all():
                raise PriceError(f"{name} is not finite in weeks {_weeks(~finite)}")
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        flat = self.slope_eur_mwh_per_gwh <= 0
        if flat.any():
            raise PriceError(
                "the price slope must be > 0 in every stage (otherwise the cost is not convex); "
                f"it is not in weeks {_weeks(flat)}"
            )

    def price_eur_mwh(self, net_import_gwh: ArrayLike, week: ArrayLike) -> np.ndarray:
        """Price at a given net import: ``p_t(N) = a_t + b_t N``.

        Parameters
        ----------
        net_import_gwh : array_like
            Net import [GWh per 168-hour stage], any shape.
        week : int or array_like of int
            ISO week(s) 1-52, broadcast against ``net_import_gwh``.

        Returns
        -------
        numpy.ndarray
            Price [EUR/MWh].
        """
        index = _stage_index(week)
        net_import = np.asarray(net_import_gwh, dtype=float)
        return self.intercept_eur_mwh[index] + self.slope_eur_mwh_per_gwh[index] * net_import

    def cost_keur(self, net_import_gwh: ArrayLike, week: ArrayLike) -> np.ndarray:
        """Cost of covering a net import: ``C_t(N) = a_t N + b_t / 2 N^2``.

        The area under the price curve from 0 to ``N``; its derivative is
        :meth:`price_eur_mwh`. Negative for exports (``N < 0``) when ``a_t > 0``: revenue.

        Parameters
        ----------
        net_import_gwh : array_like
            Net import [GWh per 168-hour stage], any shape (e.g. the SDP's ``(K, J, K)`` grid).
        week : int or array_like of int
            ISO week(s) 1-52, broadcast against ``net_import_gwh``.

        Returns
        -------
        numpy.ndarray
            Cost [kEUR]: GWh x EUR/MWh = kEUR.
        """
        index = _stage_index(week)
        net_import = np.asarray(net_import_gwh, dtype=float)
        intercept = self.intercept_eur_mwh[index]
        slope = self.slope_eur_mwh_per_gwh[index]
        energy_cost = intercept * net_import + slope / 2 * net_import**2  # [GWh EUR/MWh]
        return energy_cost * MWH_PER_GWH / EUR_PER_KEUR


@dataclass(frozen=True)
class PriceCurve:
    """Fitted price curve of one variant: coefficients, HAC standard errors and fit statistics.

    ``params`` and ``std_errors`` are keyed by coefficient (:attr:`CurveSpec.terms`): ``alpha``
    [EUR/MWh], ``season_<name>`` [EUR/MWh], ``gamma`` [-], and ``beta`` [EUR/MWh per GWh] or
    ``kappa`` [1/GWh]. ``resid_std_eur_mwh`` is the residual standard error [EUR/MWh].
    """

    spec: CurveSpec
    params: dict[str, float]
    std_errors: dict[str, float]
    hac_maxlags: int
    nobs: int
    years: tuple[int, ...]
    rsquared: float
    rsquared_adj: float
    resid_std_eur_mwh: float
    durbin_watson: float
    bic: float

    @property
    def alpha_eur_mwh(self) -> float:
        """Intercept of the reference season [EUR/MWh]."""
        return self.params[ALPHA]

    @property
    def gamma(self) -> float:
        """Pass-through of the German price [-]."""
        return self.params[GAMMA]

    @property
    def slope_coefficient(self) -> float:
        """``beta`` [EUR/MWh per GWh] or ``kappa`` [1/GWh], by the slope form."""
        return self.params[self.spec.slope_term]

    @property
    def season_effects_eur_mwh(self) -> dict[str, float]:
        """Intercept shift of each season, 0 for the reference [EUR/MWh]; empty without seasons."""
        names = [name for name, _ in self.spec.seasons]
        return {name: self.params.get(SEASON_PREFIX + name, 0.0) for name in names}

    @property
    def t_values(self) -> dict[str, float]:
        """HAC t-value of each coefficient [-]."""
        return {name: self.params[name] / self.std_errors[name] for name in self.params}

    @property
    def kappa_per_gwh(self) -> float | None:
        """``kappa`` of the proportional form [1/GWh]; None for the constant form."""
        return self.params[KAPPA] if self.spec.slope_form == "proportional" else None

    def de_pass_through(self, net_import_gwh: ArrayLike) -> np.ndarray:
        """Change of the Swedish price per unit change of the German price, at a net import.

        ``dp/dp_DE = gamma + kappa N`` for the proportional form and ``gamma`` for the constant
        form.

        Parameters
        ----------
        net_import_gwh : array_like
            Net import [GWh per 168-hour stage].

        Returns
        -------
        numpy.ndarray
            Pass-through [EUR/MWh per EUR/MWh = -], shaped like ``net_import_gwh``.
        """
        net_import = np.asarray(net_import_gwh, dtype=float)
        kappa = self.kappa_per_gwh if self.kappa_per_gwh is not None else 0.0
        return self.gamma + kappa * net_import

    def predict_eur_mwh(self, table: pd.DataFrame) -> pd.Series:
        """Price predicted for each row of a table like :func:`price_table` [EUR/MWh].

        Rows with a missing input give NaN.
        """
        x = design_matrix(table, self.spec)
        return x @ pd.Series(self.params)[list(x.columns)]

    def stage_curve(self, de_price_eur_mwh: ArrayLike) -> StageCurve:
        """Weekly intercept and slope of this curve for a German price profile.

        Parameters
        ----------
        de_price_eur_mwh : array_like
            German price of each stage, ISO weeks 1-52 [EUR/MWh] (see :func:`de_price_profile`).

        Returns
        -------
        StageCurve

        Raises
        ------
        PriceError
            If the profile does not have 52 finite values, or a slope is <= 0 (with the
            proportional form, any week with a German price <= 0).
        """
        price_de = np.asarray(de_price_eur_mwh, dtype=float)
        if price_de.shape != (MODEL_WEEKS_PER_YEAR,):
            raise PriceError(
                f"the German price profile needs {MODEL_WEEKS_PER_YEAR} stages, got shape "
                f"{price_de.shape}"
            )
        weeks = np.arange(1, MODEL_WEEKS_PER_YEAR + 1)
        season = np.zeros(len(weeks))
        if self.spec.seasons:
            effects = self.season_effects_eur_mwh
            season = np.array([effects[s] for s in season_of_week(weeks, self.spec.seasons)])
        intercept = self.alpha_eur_mwh + season + self.gamma * price_de
        if self.spec.slope_form == "constant":
            slope = np.full(len(weeks), self.slope_coefficient)
        else:
            slope = self.slope_coefficient * price_de
        return StageCurve(intercept, slope, price_de)

    def to_dict(self) -> dict:
        """JSON-ready form, with the unit of every coefficient."""
        units = {
            name: UNITS[SEASON_PREFIX if name.startswith(SEASON_PREFIX) else name]
            for name in self.params
        }
        return {
            "spec": self.spec.to_dict(),
            "params": self.params,
            "std_errors": self.std_errors,
            "units": units,
            "hac_maxlags": self.hac_maxlags,
            "nobs": self.nobs,
            "years": list(self.years),
            "rsquared": self.rsquared,
            "rsquared_adj": self.rsquared_adj,
            "resid_std_eur_mwh": self.resid_std_eur_mwh,
            "durbin_watson": self.durbin_watson,
            "bic": self.bic,
        }

    @classmethod
    def from_dict(cls, raw: Mapping) -> PriceCurve:
        """Inverse of :meth:`to_dict`; checks the coefficients, their units and the slope.

        Raises
        ------
        PriceError
            If the coefficients or standard errors do not match the variant's terms, a stated
            unit differs from :data:`UNITS`, or the slope is <= 0.
        """
        spec = CurveSpec.from_dict(raw["spec"])
        params = {name: float(value) for name, value in raw["params"].items()}
        std_errors = {name: float(value) for name, value in raw["std_errors"].items()}
        for what, names in (("coefficients", params), ("standard errors", std_errors)):
            if set(names) != set(spec.terms):
                raise PriceError(f"{what} {sorted(names)} do not match {spec.label}")
        expected_units = {
            name: UNITS[SEASON_PREFIX if name.startswith(SEASON_PREFIX) else name]
            for name in spec.terms
        }
        if raw.get("units", expected_units) != expected_units:
            raise PriceError(f"units {raw['units']} differ from {expected_units}")
        curve = cls(
            spec=spec,
            params=params,
            std_errors=std_errors,
            hac_maxlags=int(raw["hac_maxlags"]),
            nobs=int(raw["nobs"]),
            years=tuple(int(year) for year in raw["years"]),
            rsquared=float(raw["rsquared"]),
            rsquared_adj=float(raw["rsquared_adj"]),
            resid_std_eur_mwh=float(raw["resid_std_eur_mwh"]),
            durbin_watson=float(raw["durbin_watson"]),
            bic=float(raw["bic"]),
        )
        _check_slope(curve)
        return curve


def _check_slope(curve: PriceCurve) -> None:
    """Stop on a slope coefficient <= 0 (the guide's rule): a sign, unit or alignment error."""
    if not curve.slope_coefficient > 0:
        raise PriceError(
            f"{curve.spec.label}: slope {curve.spec.slope_term} = {curve.slope_coefficient:.6g} "
            "is not > 0; check the sign, units and alignment of net import and prices"
        )


def fit_curve(table: pd.DataFrame, spec: CurveSpec, hac_maxlags: int) -> PriceCurve:
    """Fit one variant of the price curve by OLS, with HAC standard errors.

    The Newey-West standard errors and the Durbin-Watson statistic treat the fitted weeks as one
    series: a dropped week, or an excluded block of years, joins the weeks on either side as
    neighbours. Only the reported standard errors of the variants without the crisis years
    are affected noticeably; the coefficients and predictions are not.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`price_table`; incomplete weeks are left out.
    spec : CurveSpec
        The variant (slope form, seasons, excluded years).
    hac_maxlags : int
        Lags of the Newey-West standard errors [weeks] (Bartlett kernel, no small-sample
        correction).

    Returns
    -------
    PriceCurve

    Raises
    ------
    PriceError
        If an excluded year is not in the data, the design is rank-deficient, or the slope
        coefficient is <= 0.
    """
    results, used = _ols(table, spec, hac_maxlags)
    curve = PriceCurve(
        spec=spec,
        params={name: float(value) for name, value in results.params.items()},
        std_errors={name: float(value) for name, value in results.bse.items()},
        hac_maxlags=hac_maxlags,
        nobs=int(results.nobs),
        years=tuple(sorted(int(year) for year in table.loc[used, "iso_year"].unique())),
        rsquared=float(results.rsquared),
        rsquared_adj=float(results.rsquared_adj),
        resid_std_eur_mwh=float(np.sqrt(results.scale)),
        durbin_watson=float(durbin_watson(results.resid)),
        bic=float(results.bic),
    )
    _check_slope(curve)
    return curve


def loyo_predictions(table: pd.DataFrame, spec: CurveSpec, hac_maxlags: int) -> pd.Series:
    """Leave-one-year-out price of every complete week [EUR/MWh].

    Each ISO year is predicted by the variant refitted without that year, so no prediction
    has seen its own year. Years the variant excludes anyway are predicted by its own fit.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`price_table`.
    spec : CurveSpec
        The variant.
    hac_maxlags : int
        Lags of the Newey-West standard errors [weeks] (the refits check their slope too).

    Returns
    -------
    pandas.Series
        Indexed like ``table``; NaN for incomplete weeks.
    """
    out = pd.Series(np.nan, index=table.index, name="loyo_eur_mwh")
    complete = table["complete"].to_numpy()
    years = table["iso_year"].to_numpy()
    for year in np.unique(years[complete]):
        rows = complete & (years == year)
        curve = fit_curve(table, without_year(spec, int(year)), hac_maxlags)
        out[rows] = curve.predict_eur_mwh(table[rows]).to_numpy()
    return out


def de_price_profile(
    table: pd.DataFrame,
    years: Sequence[int] | None = None,
    level_eur_mwh: float | None = None,
) -> np.ndarray:
    """Mean German price of each ISO week over the years [EUR/MWh].

    Parameters
    ----------
    table : pandas.DataFrame
        With ``iso_year``, ``iso_week`` and ``price_de_eur_mwh`` [EUR/MWh]; missing prices are
        skipped.
    years : sequence of int or None
        ISO years to average; all when None.
    level_eur_mwh : float or None
        If given, the profile is rescaled so that its mean over the 52 stages equals this level
        (the continental price scenarios) [EUR/MWh].

    Returns
    -------
    numpy.ndarray
        Shape ``(52,)``, position = ISO week - 1 [EUR/MWh].

    Raises
    ------
    PriceError
        If a week has no German price in any of the years.
    """
    rows = table if years is None else table[table["iso_year"].isin(list(years))]
    weeks = range(1, MODEL_WEEKS_PER_YEAR + 1)
    profile = rows.groupby("iso_week")["price_de_eur_mwh"].mean().reindex(weeks)
    if profile.isna().any():
        raise PriceError(f"no German price in weeks {profile.index[profile.isna()].tolist()}")
    values = profile.to_numpy(dtype=float)
    if level_eur_mwh is not None:
        values = values * (level_eur_mwh / values.mean())
    return values


@dataclass(frozen=True, eq=False)
class Variant:
    """A fitted variant and its leave-one-year-out prices [EUR/MWh]."""

    curve: PriceCurve
    loyo_eur_mwh: pd.Series


def variant_specs(cfg: AppConfig) -> list[CurveSpec]:
    """List the variants: slope forms x with/without seasons x with/without crisis years."""
    pm = cfg.price_model
    seasons = tuple(pm.seasons.items())
    samples = dict.fromkeys([(), tuple(sorted(pm.crisis_years))])  # one sample if no crisis
    return [
        CurveSpec(form, seasons if seasonal else (), sample)
        for form in SLOPE_FORMS
        for seasonal in (False, True)
        for sample in samples
    ]


def base_spec(cfg: AppConfig) -> CurveSpec:
    """Return the configured base model (``price_model.slope_form``, ``seasonal_dummies``)."""
    pm = cfg.price_model
    seasons = tuple(pm.seasons.items()) if pm.seasonal_dummies else ()
    return CurveSpec(pm.slope_form, seasons, pm.exclude_years)


def _rmse(errors: pd.Series) -> float:
    """Root mean square of the non-missing errors [EUR/MWh]; NaN if there are none."""
    valid = errors.dropna()
    return float(np.sqrt((valid**2).mean())) if len(valid) else float("nan")


def _periods(years: pd.Series, crisis_years: Sequence[int]) -> dict[str, pd.Series]:
    """Masks of the years before, during and after the crisis years."""
    if not crisis_years:
        return {}
    first, last = min(crisis_years), max(crisis_years)
    return {
        "before": years < first,
        "crisis": years.isin(list(crisis_years)),
        "after": years > last,
    }


def loyo_accuracy(
    errors: pd.Series, season: pd.Series, periods: Mapping[str, pd.Series]
) -> dict[str, float]:
    """Out-of-sample accuracy of one variant from its LOYO errors.

    Parameters
    ----------
    errors : pandas.Series
        Observed minus leave-one-year-out price [EUR/MWh]; missing values are skipped.
    season : pandas.Series
        Season of each week, aligned with ``errors``.
    periods : mapping of str to pandas.Series
        Boolean masks of sub-periods (e.g. before, during and after the crisis years).

    Returns
    -------
    dict of str to float
        ``loyo_rmse_eur_mwh`` (pooled), ``loyo_rmse_<period>_eur_mwh`` for each period and
        ``loyo_season_bias_eur_mwh``, the largest absolute mean error of a season [EUR/MWh].
    """
    return {
        "loyo_rmse_eur_mwh": _rmse(errors),
        **{f"loyo_rmse_{name}_eur_mwh": _rmse(errors[mask]) for name, mask in periods.items()},
        "loyo_season_bias_eur_mwh": float(errors.groupby(season).mean().abs().max()),
    }


def compare_variants(
    table: pd.DataFrame, variants: Mapping[str, Variant], crisis_years: Sequence[int]
) -> pd.DataFrame:
    """Coefficients, fit statistics and out-of-sample accuracy of every variant.

    The out-of-sample errors (observed minus LOYO price) must cover the same weeks for every
    variant, so their RMSEs compare; in-sample R2 does not, across different samples.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`price_table`.
    variants : mapping of str to Variant
        Fitted variants by label.
    crisis_years : sequence of int
        Years that split the period into before, crisis and after for the RMSE by period.

    Returns
    -------
    pandas.DataFrame
        Indexed by label: ``slope_form``, ``seasonal``, ``excluded``, ``n_obs``, ``n_params``,
        the coefficients (NaN where absent), ``slope_t_hac``, ``rsquared``, ``rsquared_adj``,
        ``resid_std_eur_mwh``, ``durbin_watson``, ``bic``, ``loyo_rmse_eur_mwh`` (pooled) and
        per period (``loyo_rmse_before_eur_mwh`` etc.) [EUR/MWh], and
        ``loyo_season_bias_eur_mwh``: the largest absolute mean LOYO error of a season [EUR/MWh].

    Raises
    ------
    PriceError
        If the variants' LOYO predictions do not cover the same weeks.
    """
    scored = {label: v.loyo_eur_mwh.notna().to_numpy() for label, v in variants.items()}
    reference = next(iter(scored.values()))
    differing = [label for label, mask in scored.items() if not np.array_equal(mask, reference)]
    if differing:
        raise PriceError(f"variants scored on other weeks than the first: {differing}")
    price = table["price_se_eur_mwh"]
    periods = _periods(table["iso_year"], crisis_years)
    terms = list(dict.fromkeys(t for v in variants.values() for t in v.curve.spec.terms))
    seasons = [t for t in terms if t.startswith(SEASON_PREFIX)]
    terms = [ALPHA, *seasons, GAMMA, BETA, KAPPA]
    rows = []
    for label, variant in variants.items():
        curve, spec = variant.curve, variant.curve.spec
        errors = price - variant.loyo_eur_mwh
        rows.append(
            {
                "label": label,
                "slope_form": spec.slope_form,
                "seasonal": bool(spec.seasons),
                "excluded": _year_span(spec.exclude_years),
                "n_obs": curve.nobs,
                "n_params": len(curve.params),
                **{term: curve.params.get(term, np.nan) for term in terms},
                "slope_t_hac": curve.t_values[spec.slope_term],
                "rsquared": curve.rsquared,
                "rsquared_adj": curve.rsquared_adj,
                "resid_std_eur_mwh": curve.resid_std_eur_mwh,
                "durbin_watson": curve.durbin_watson,
                "bic": curve.bic,
                **loyo_accuracy(errors, table["season"], periods),
            }
        )
    return pd.DataFrame(rows).set_index("label")


def select_variant(comparison: pd.DataFrame, tolerance_eur_mwh: float) -> str:
    """Label of the variant chosen by the selection rule.

    1. The lowest pooled LOYO RMSE; variants within ``tolerance_eur_mwh`` of it tie.
    2. Among tied variants, the smallest LOYO seasonal bias: the SDP moves water between
       seasons, so a seasonal bias in the cost curve matters more than a sliver of RMSE.
    3. Then the fewest parameters (and then the lower RMSE).

    Parameters
    ----------
    comparison : pandas.DataFrame
        Output of :func:`compare_variants` (``loyo_rmse_eur_mwh``,
        ``loyo_season_bias_eur_mwh``, ``n_params``), indexed by label.
    tolerance_eur_mwh : float
        RMSE difference below which variants tie [EUR/MWh].

    Returns
    -------
    str
    """
    rmse = comparison["loyo_rmse_eur_mwh"]
    tied = comparison[rmse <= rmse.min() + tolerance_eur_mwh]
    ranked = tied.sort_values(
        ["loyo_season_bias_eur_mwh", "n_params", "loyo_rmse_eur_mwh"], kind="stable"
    )
    return str(ranked.index[0])


def slope_by_year(table: pd.DataFrame, hac_maxlags: int) -> pd.DataFrame:
    """Slope of each ISO year fitted alone (constant slope, no seasons): a diagnostic.

    If the slope scales with the price level, ``beta / mean German price`` is steadier across
    years than ``beta``; that ratio is comparable to ``kappa`` of the proportional form. A
    non-positive yearly slope is reported, not raised.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`price_table`.
    hac_maxlags : int
        Lags of the Newey-West standard errors [weeks].

    Returns
    -------
    pandas.DataFrame
        Indexed by ``iso_year``: ``beta_eur_mwh_per_gwh``, ``beta_se_eur_mwh_per_gwh`` (HAC),
        ``price_de_mean_eur_mwh``, ``beta_per_price_de_per_gwh`` [1/GWh] and ``nobs``.
    """
    spec = CurveSpec("constant")
    rows = []
    for year, group in table[table["complete"]].groupby("iso_year"):
        results = sm.OLS(group["price_se_eur_mwh"], design_matrix(group, spec)).fit(
            cov_type="HAC", cov_kwds={"maxlags": hac_maxlags}
        )
        mean_price_de = float(group["price_de_eur_mwh"].mean())
        rows.append(
            {
                "iso_year": int(year),
                "beta_eur_mwh_per_gwh": float(results.params[BETA]),
                "beta_se_eur_mwh_per_gwh": float(results.bse[BETA]),
                "price_de_mean_eur_mwh": mean_price_de,
                "beta_per_price_de_per_gwh": float(results.params[BETA]) / mean_price_de,
                "nobs": int(results.nobs),
            }
        )
    return pd.DataFrame(rows).set_index("iso_year")


def hac_sensitivity(table: pd.DataFrame, spec: CurveSpec, lags: Sequence[int]) -> pd.Series:
    """HAC t-value of the slope coefficient for several lag lengths.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`price_table`.
    spec : CurveSpec
        The variant.
    lags : sequence of int
        Newey-West lag lengths [weeks].

    Returns
    -------
    pandas.Series
        t-value [-] indexed by the lag length.
    """
    values = {int(lag): float(_ols(table, spec, lag)[0].tvalues[spec.slope_term]) for lag in lags}
    return pd.Series(values, name="slope_t_hac").rename_axis("hac_maxlags")


def residual_acf(residuals: pd.Series, lags: Sequence[int]) -> pd.Series:
    """Autocorrelation of a residual series at the given lags [-].

    Missing values are dropped first, so a week on each side of a gap counts as neighbours.
    """
    values = residuals.dropna().to_numpy(dtype=float)
    centred = values - values.mean()
    variance = float(centred @ centred)
    acf = {
        int(lag): float(centred[lag:] @ centred[: len(centred) - lag]) / variance for lag in lags
    }
    return pd.Series(acf, name="acf").rename_axis("lag")


@dataclass(frozen=True, eq=False)
class PriceResult:
    """Calibrated price curve, the compared variants and the report.

    ``table`` is :func:`price_table` plus the base model's ``status`` (used, missing or
    excluded), ``fitted_eur_mwh``, ``residual_eur_mwh``, ``loyo_eur_mwh`` and
    ``loyo_residual_eur_mwh`` [EUR/MWh]. ``stages`` holds the base model's weekly curve for the
    average German price profile of the period (``"average 2015-2025"``) and for that profile
    scaled to each scenario level (``"scaled to 50 EUR/MWh"`` ...).
    """

    table: pd.DataFrame
    base: PriceCurve
    variants: dict[str, Variant]
    comparison: pd.DataFrame
    selected: str
    slopes_by_year: pd.DataFrame
    stages: dict[str, StageCurve]
    report: list[str]


def _with_base_columns(table: pd.DataFrame, variant: Variant) -> pd.DataFrame:
    """Add the base model's status, fitted, residual and LOYO columns to the table."""
    out = table.copy()
    spec = variant.curve.spec
    complete = out["complete"].to_numpy()
    excluded = out["iso_year"].isin(spec.exclude_years).to_numpy()
    out["status"] = np.select([~complete, excluded], [MISSING, EXCLUDED], default=USED)
    price = out["price_se_eur_mwh"]
    out["fitted_eur_mwh"] = variant.curve.predict_eur_mwh(out)
    out["residual_eur_mwh"] = price - out["fitted_eur_mwh"]
    out["loyo_eur_mwh"] = variant.loyo_eur_mwh
    out["loyo_residual_eur_mwh"] = price - out["loyo_eur_mwh"]
    return out


def _format_table(frame: pd.DataFrame, decimals: Mapping[str, int]) -> str:
    """Format a report table with a fixed number of decimals per column."""
    shown = frame[[column for column in decimals if column in frame.columns]]
    return shown.round(dict(decimals)).to_string()


def _coefficient_lines(curve: PriceCurve) -> list[str]:
    lines = [f"  {'coefficient':<12} {'estimate':>12} {'HAC SE':>12} {'t':>8}  unit"]
    t_values = curve.t_values
    for name, value in curve.params.items():
        unit = UNITS[SEASON_PREFIX if name.startswith(SEASON_PREFIX) else name]
        lines.append(
            f"  {name:<12} {value:>12.5g} {curve.std_errors[name]:>12.3g} "
            f"{t_values[name]:>8.1f}  {unit}"
        )
    return lines


def _residual_lines(table: pd.DataFrame, cfg: AppConfig) -> list[str]:
    used = table[table["status"] == USED]
    by_season = pd.DataFrame(
        {
            "mean": used.groupby("season")["residual_eur_mwh"].mean(),
            "std": used.groupby("season")["residual_eur_mwh"].std(),
            "loyo_mean": used.groupby("season")["loyo_residual_eur_mwh"].mean(),
            "loyo_rmse": used.groupby("season")["loyo_residual_eur_mwh"].agg(_rmse),
        }
    ).reindex(list(cfg.price_model.seasons))
    by_year = pd.DataFrame(
        {
            "mean": used.groupby("iso_year")["residual_eur_mwh"].mean(),
            "std": used.groupby("iso_year")["residual_eur_mwh"].std(),
            "loyo_mean": used.groupby("iso_year")["loyo_residual_eur_mwh"].mean(),
            "loyo_rmse": used.groupby("iso_year")["loyo_residual_eur_mwh"].agg(_rmse),
        }
    )
    acf = residual_acf(used["residual_eur_mwh"], cfg.price_model.residual_acf_lags)
    acf_text = ", ".join(f"lag {lag}: {value:.2f}" for lag, value in acf.items())
    return [
        "\nResiduals of the base model (observed - model) [EUR/MWh]; loyo = out of sample",
        "  by season",
        by_season.round(1).to_string(),
        "  by ISO year",
        by_year.round(1).to_string(),
        f"  autocorrelation of the in-sample residuals [weeks]: {acf_text}",
    ]


def _stage_lines(stages: Mapping[str, StageCurve], table: pd.DataFrame) -> list[str]:
    net_import = table.loc[table["complete"], "net_import_stage_gwh"]
    lines = [
        "\nWeekly curve of the base model, a_t [EUR/MWh] and b_t [EUR/MWh per GWh] over 52 stages"
    ]
    for name, stage in stages.items():
        a, b = stage.intercept_eur_mwh, stage.slope_eur_mwh_per_gwh
        lines.append(
            f"  {name}: German price mean {stage.de_price_eur_mwh.mean():.1f} EUR/MWh "
            f"(weeks {stage.de_price_eur_mwh.min():.1f} to {stage.de_price_eur_mwh.max():.1f}); "
            f"a_t {a.min():.1f} to {a.max():.1f}; b_t {b.min():.4f} to {b.max():.4f}"
        )
    lines.append(
        f"  historical net import per stage: {net_import.min():.0f} to {net_import.max():.0f} GWh "
        f"(mean {net_import.mean():.0f})"
    )
    return lines


def price_report(
    cfg: AppConfig,
    table: pd.DataFrame,
    base: PriceCurve,
    comparison: pd.DataFrame,
    selected: str,
    slopes: pd.DataFrame,
    stages: Mapping[str, StageCurve],
) -> list[str]:
    """Human-readable summary: sample, base coefficients, variants, selection, diagnostics.

    Parameters
    ----------
    cfg : AppConfig
        Configuration (``price_model``).
    table : pandas.DataFrame
        The table of :class:`PriceResult`.
    base : PriceCurve
        The configured base model.
    comparison : pandas.DataFrame
        Output of :func:`compare_variants` with the ``selected`` and ``configured_base`` flags.
    selected : str
        Label chosen by :func:`select_variant`.
    slopes : pandas.DataFrame
        Output of :func:`slope_by_year`.
    stages : mapping of str to StageCurve
        The base model's weekly curve per German price profile.

    Returns
    -------
    list of str
        Report lines; prices in EUR/MWh, net import in GWh per 168-hour stage.
    """
    pm = cfg.price_model
    firsts = list(pm.seasons.values())
    lasts = [first - 1 for first in firsts[1:]] + [MODEL_WEEKS_PER_YEAR]
    season_text = ", ".join(
        f"{name} weeks {first}-{last}"
        for name, first, last in zip(pm.seasons, firsts, lasts, strict=True)
    )
    lines = [
        "Price curve p = alpha + season + gamma p_DE + slope N [EUR/MWh]; N = net import per "
        "168-h stage [GWh] (residual demand - hydro; negative when Sweden exports)",
        "  slope: constant = beta [EUR/MWh per GWh] | proportional = kappa p_DE, kappa [1/GWh]",
        "  cost of covering N in stage t: C_t(N) = a_t N + b_t/2 N^2 [kEUR], "
        "a_t = alpha + season_t + gamma p_DE,t",
        f"  OLS; HAC (Newey-West, Bartlett) standard errors with {pm.hac_maxlags} lags [weeks]; "
        "the weeks on either side of a gap or of excluded years count as neighbours",
        f"  seasons: {season_text} (the first is the reference)",
    ]

    dropped = table[~table["complete"]]
    years = sorted(table["iso_year"].unique())
    lines.append(
        f"\nSample: {int(table['complete'].sum())} of {len(table)} weeks complete "
        f"({years[0]}-{years[-1]}); weeks left out for missing inputs: {len(dropped)}"
    )
    for y, k, missing in zip(
        dropped["iso_year"], dropped["iso_week"], dropped["missing_inputs"], strict=True
    ):
        lines.append(f"  {week_label(y, k)}: {missing}")

    errors = table["loyo_residual_eur_mwh"]
    lines += [
        f"\nBase model (configured): {base.spec.label}",
        f"  n = {base.nobs} weeks, ISO years {_year_span(base.years)}",
        *_coefficient_lines(base),
        f"  R2 {base.rsquared:.3f}, adjusted R2 {base.rsquared_adj:.3f}, residual std "
        f"{base.resid_std_eur_mwh:.1f} EUR/MWh, Durbin-Watson {base.durbin_watson:.2f}, "
        f"BIC {base.bic:.0f}",
    ]
    sensitivity = hac_sensitivity(table, base.spec, pm.hac_lag_sensitivity)
    lines.append(
        f"  slope t-value ({base.spec.slope_term}) by HAC lags [weeks]: "
        + ", ".join(f"{lag}: {t:.1f}" for lag, t in sensitivity.items())
    )
    lines.append(
        f"  LOYO RMSE {_rmse(errors):.1f} EUR/MWh; LOYO seasonal bias "
        f"{comparison.loc[base.spec.label, 'loyo_season_bias_eur_mwh']:.1f} EUR/MWh"
    )

    lines.append(
        "\nVariants: fit and accuracy. LOYO RMSE = leave-one-year-out RMSE of the weekly price "
        "[EUR/MWh], pooled over all complete weeks and by period (before, during and after "
        f"{_year_span(pm.crisis_years)}); season bias = largest |mean LOYO error| of a season"
    )
    accuracy = {
        "n_obs": 0,
        "n_params": 0,
        "slope_t_hac": 1,
        "rsquared": 3,
        "durbin_watson": 2,
        "loyo_rmse_eur_mwh": 2,
        "loyo_rmse_before_eur_mwh": 1,
        "loyo_rmse_crisis_eur_mwh": 1,
        "loyo_rmse_after_eur_mwh": 1,
        "loyo_season_bias_eur_mwh": 2,
    }
    lines.append(_format_table(comparison, accuracy))
    lines.append("\nVariants: coefficients (units as above)")
    coefficients = {column: 4 for column in comparison.columns if column.startswith(SEASON_PREFIX)}
    lines.append(_format_table(comparison, {ALPHA: 2, **coefficients, GAMMA: 3, BETA: 4, KAPPA: 6}))

    agrees = "yes" if selected == base.spec.label else "NO"
    lines += [
        "\nSelection rule: the lowest pooled LOYO RMSE; variants within "
        f"{pm.selection_tolerance_eur_mwh:g} EUR/MWh of it tie, and the smallest LOYO seasonal "
        "bias wins, then the fewest parameters",
        f"  selected by the rule: {selected}",
        f"  configured base model: {base.spec.label} (agrees with the rule: {agrees})",
    ]

    lines += _residual_lines(table, cfg)
    lines.append(
        "\nSlope of each ISO year fitted alone (constant slope, no seasons): beta "
        "[EUR/MWh per GWh], mean German price [EUR/MWh], beta / mean German price [1/GWh]"
    )
    lines.append(
        slopes.round(
            {
                "beta_eur_mwh_per_gwh": 4,
                "beta_se_eur_mwh_per_gwh": 4,
                "price_de_mean_eur_mwh": 1,
                "beta_per_price_de_per_gwh": 6,
            }
        ).to_string()
    )
    lines += _stage_lines(stages, table)
    return lines


def build_price(cfg: AppConfig, weekly: pd.DataFrame) -> PriceResult:
    """Fit every variant, compare them out of sample and report on the configured base model.

    Parameters
    ----------
    cfg : AppConfig
        Configuration with the ``period``, ``price_model`` and ``scenarios`` sections.
    weekly : pandas.DataFrame
        Weekly dataset (``weekly_se.csv`` contract) covering the ISO years of ``cfg.period``.

    Returns
    -------
    PriceResult

    Raises
    ------
    PriceError
        If the years differ from the period, a fit has a slope <= 0 (also a leave-one-year-out
        refit), or the base model's weekly slope is <= 0 for the average German price profile
        or a scenario level.
    """
    pm = cfg.price_model
    table = price_table(weekly, pm.seasons)
    first, last = cfg.period.first_iso_year, cfg.period.last_iso_year
    years = sorted(int(year) for year in table["iso_year"].unique())
    if years != list(range(first, last + 1)):
        raise PriceError(
            f"the weekly dataset covers the ISO years {years}, not the configured period "
            f"{first}-{last}: rebuild weekly_se.csv"
        )
    base = base_spec(cfg)
    base_curve = fit_curve(table, base, pm.hac_maxlags)  # first, so a failure names the base
    variants: dict[str, Variant] = {}
    for spec in variant_specs(cfg):
        curve = base_curve if spec == base else fit_curve(table, spec, pm.hac_maxlags)
        variants[spec.label] = Variant(curve, loyo_predictions(table, spec, pm.hac_maxlags))
    comparison = compare_variants(table, variants, pm.crisis_years)
    selected = select_variant(comparison, pm.selection_tolerance_eur_mwh)
    comparison["selected"] = comparison.index == selected
    comparison["configured_base"] = comparison.index == base.label

    stages = {f"average {first}-{last}": base_curve.stage_curve(de_price_profile(table))}
    for level in cfg.scenarios.de_price_levels_eur_mwh:
        profile = de_price_profile(table, level_eur_mwh=level)
        stages[f"scaled to {level:g} EUR/MWh"] = base_curve.stage_curve(profile)
    table = _with_base_columns(table, variants[base.label])
    slopes = slope_by_year(table, pm.hac_maxlags)
    report = price_report(cfg, table, base_curve, comparison, selected, slopes, stages)
    return PriceResult(table, base_curve, variants, comparison, selected, slopes, stages, report)


def save_curve(curve: PriceCurve, path: Path) -> None:
    """Write a fitted curve as JSON at full precision (read it back with :func:`load_curve`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(curve.to_dict(), indent=2)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")


def load_curve(path: Path) -> PriceCurve:
    """Read a curve written by :func:`save_curve` and check it.

    Raises
    ------
    PriceError
        If the coefficients do not match the variant or the slope is <= 0.
    """
    return PriceCurve.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def write_outputs(cfg: AppConfig, result: PriceResult) -> None:
    """Write the weekly table, the variant comparison, the base curve and the report.

    Paths come from ``cfg.price_model``. The comparison keeps full precision (``kappa`` is of
    the order of 1e-3).
    """
    pm = cfg.price_model
    write_weekly_csv(result.table, PROJECT_ROOT / pm.weekly_csv, cfg.data.csv_float_format)
    variants_path = PROJECT_ROOT / pm.variants_csv
    variants_path.parent.mkdir(parents=True, exist_ok=True)
    result.comparison.to_csv(variants_path, lineterminator="\n", encoding="utf-8")
    save_curve(result.base, PROJECT_ROOT / pm.curve_json)
    report_path = PROJECT_ROOT / pm.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(result.report) + "\n", encoding="utf-8", newline="\n")
