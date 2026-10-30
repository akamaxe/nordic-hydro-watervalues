"""Tests for the price curve and the cost function (synthetic data, offline)."""

import json
from pathlib import Path

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
from watervalues.price import (
    REGRESSION_INPUTS,
    CurveSpec,
    PriceError,
    StageCurve,
    Variant,
    build_price,
    compare_variants,
    de_price_profile,
    fit_curve,
    load_curve,
    loyo_accuracy,
    loyo_predictions,
    price_table,
    save_curve,
    select_variant,
)
from watervalues.units import HOURS_PER_WEEK, MODEL_WEEKS_PER_YEAR

CFG = load_config()
PM = CFG.price_model
SEASONS = PM.seasons
LAGS = PM.hac_maxlags
FIRST, LAST = 2019, 2021  # 2020 has an ISO week 53, folded into week 52 (336 h)
SYNTHETIC_CFG = CFG.model_copy(
    update={
        "period": PeriodConfig(first_iso_year=FIRST, last_iso_year=LAST),
        "price_model": PM.model_copy(update={"crisis_years": (LAST,), "exclude_years": ()}),
    }
)
# True parameters: alpha [EUR/MWh], gamma [-], beta [EUR/MWh per GWh], kappa [1/GWh]
ALPHA, GAMMA, BETA, KAPPA = 5.0, 0.9, 0.04, 0.0008
SEASON_EFFECTS = dict(zip(list(SEASONS)[1:], (-2.0, -6.0, 3.0), strict=True))  # [EUR/MWh]
SLOPE = {"constant": BETA, "proportional": KAPPA}
# Non-hydro generation per 168-hour stage [GWh]; residual demand = 3000 - 1700 = 1300 GWh
LOAD, NUCLEAR, WIND, SOLAR, OTHER = 3_000.0, 1_000.0, 500.0, 50.0, 150.0


def synthetic_weekly(
    net_import_gwh: np.ndarray, price_de_eur_mwh: np.ndarray, price_se_eur_mwh: np.ndarray
) -> pd.DataFrame:
    """Weekly dataset with the given net import per 168-hour stage and the given prices.

    Energies are totals over each row's hours, so the folded week 52 (336 h) holds twice the
    per-stage values: a model that forgets to rescale it cannot recover the parameters. Integer
    energies keep every sum exact in floating point.
    """
    cal = week_calendar(FIRST, LAST)
    rows = cal["hours"].to_numpy() / HOURS_PER_WEEK
    residual = LOAD - NUCLEAR - WIND - SOLAR - OTHER
    columns = {name: np.full(len(cal), 1.0) for name in WEEKLY_COLUMNS}
    columns.update(
        week_start=cal["week_start"].to_numpy(),
        iso_year=cal["iso_year"].to_numpy(),
        iso_week=cal["iso_week"].to_numpy(),
        hours=cal["hours"].to_numpy(),
        load_gwh=LOAD * rows,
        nuclear_gwh=NUCLEAR * rows,
        wind_gwh=WIND * rows,
        solar_gwh=SOLAR * rows,
        other_gwh=OTHER * rows,
        hydro_gwh=(residual - net_import_gwh) * rows,
        price_se_eur_mwh=price_se_eur_mwh,
        price_de_eur_mwh=price_de_eur_mwh,
    )
    return pd.DataFrame(columns)[list(WEEKLY_COLUMNS)]


def design(
    cal: pd.DataFrame, net_import: np.ndarray, price_de: np.ndarray, slope_form: str, seasonal: bool
) -> np.ndarray:
    """The regression design, built here independently of watervalues.price."""
    columns = [np.ones(len(cal))]
    if seasonal:
        season = np.searchsorted(list(SEASONS.values()), cal["iso_week"], side="right") - 1
        columns += [(season == k).astype(float) for k in range(1, len(SEASONS))]
    columns.append(price_de)
    columns.append(net_import if slope_form == "constant" else net_import * price_de)
    return np.column_stack(columns)


def truth(slope_form: str, seasonal: bool, slope: float) -> np.ndarray:
    effects = list(SEASON_EFFECTS.values()) if seasonal else []
    return np.array([ALPHA, *effects, GAMMA, slope])


@pytest.fixture
def drivers() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    cal = week_calendar(FIRST, LAST)
    rng = np.random.default_rng(3)
    net_import = rng.integers(-900, 300, len(cal)).astype(float)  # GWh per stage
    price_de = rng.integers(20, 150, len(cal)).astype(float)  # EUR/MWh
    noise = rng.normal(0.0, 4.0, len(cal))  # EUR/MWh
    return cal, net_import, price_de, noise


def synthetic_prices(
    drivers, slope_form: str, seasonal: bool, slope: float
) -> tuple[np.ndarray, np.ndarray]:
    """Swedish price X theta + e with residuals e orthogonal to X: OLS returns theta exactly."""
    cal, net_import, price_de, noise = drivers
    x = design(cal, net_import, price_de, slope_form, seasonal)
    residual = noise - x @ np.linalg.lstsq(x, noise, rcond=None)[0]
    return x @ truth(slope_form, seasonal, slope) + residual, residual


def synthetic_table(drivers, slope_form: str, seasonal: bool, slope: float) -> pd.DataFrame:
    _, net_import, price_de, _ = drivers
    price, _ = synthetic_prices(drivers, slope_form, seasonal, slope)
    return price_table(synthetic_weekly(net_import, price_de, price), SEASONS)


def spec_of(slope_form: str, seasonal: bool, exclude_years: tuple[int, ...] = ()) -> CurveSpec:
    return CurveSpec(slope_form, tuple(SEASONS.items()) if seasonal else (), exclude_years)


def week_mask(cal: pd.DataFrame, iso_year: int, iso_week: int) -> np.ndarray:
    return ((cal["iso_year"] == iso_year) & (cal["iso_week"] == iso_week)).to_numpy()


@pytest.fixture(scope="module")
def committed():
    return build_price(CFG, load_weekly(PROJECT_ROOT / CFG.data.weekly_csv))


# --- Definition of Done -------------------------------------------------------------------------


@pytest.mark.parametrize("seasonal", [False, True])
@pytest.mark.parametrize("slope_form", ["constant", "proportional"])
def test_synthetic_parameters_are_recovered_exactly(drivers, slope_form, seasonal) -> None:
    cal, net_import, price_de, _ = drivers
    price, residual = synthetic_prices(drivers, slope_form, seasonal, SLOPE[slope_form])
    table = price_table(synthetic_weekly(net_import, price_de, price), SEASONS)
    curve = fit_curve(table, spec_of(slope_form, seasonal), LAGS)
    assert curve.alpha_eur_mwh == pytest.approx(ALPHA, rel=1e-8)
    assert curve.gamma == pytest.approx(GAMMA, rel=1e-8)
    assert curve.slope_coefficient == pytest.approx(SLOPE[slope_form], rel=1e-8)
    reference = next(iter(SEASONS))
    expected = {reference: 0.0, **SEASON_EFFECTS} if seasonal else {}
    assert curve.season_effects_eur_mwh == pytest.approx(expected, abs=1e-7)
    np.testing.assert_allclose(price - curve.predict_eur_mwh(table), residual, atol=1e-7)
    assert curve.nobs == len(cal)
    assert curve.years == (FIRST, FIRST + 1, LAST)
    assert all(np.isfinite(se) and se > 0 for se in curve.std_errors.values())


@pytest.mark.parametrize("slope_form", ["constant", "proportional"])
def test_non_positive_slope_stops(drivers, slope_form) -> None:
    table = synthetic_table(drivers, slope_form, False, -SLOPE[slope_form])
    with pytest.raises(PriceError, match="slope"):
        fit_curve(table, spec_of(slope_form, False), LAGS)


def test_loyo_refits_stop_on_a_non_positive_slope(drivers) -> None:
    cal, net_import, price_de, noise = drivers
    # The slope is positive in 2021 only: positive when pooled, negative in the refit without 2021.
    slope = np.where(cal["iso_year"] == LAST, 0.3, -0.02)
    price = ALPHA + GAMMA * price_de + slope * net_import + noise
    table = price_table(synthetic_weekly(net_import, price_de, price), SEASONS)
    spec = spec_of("constant", False)
    assert fit_curve(table, spec, LAGS).slope_coefficient > 0
    with pytest.raises(PriceError, match=f"without {LAST}"):
        loyo_predictions(table, spec, LAGS)


def test_standard_errors_are_newey_west(drivers) -> None:
    cal, net_import, price_de, _ = drivers
    price, residual = synthetic_prices(drivers, "proportional", True, KAPPA)
    table = price_table(synthetic_weekly(net_import, price_de, price), SEASONS)
    curve = fit_curve(table, spec_of("proportional", True), LAGS)
    # Bartlett-kernel Newey-West covariance without small-sample correction, by hand.
    x = design(cal, net_import, price_de, "proportional", True)
    scores = x * residual[:, None]
    meat = scores.T @ scores
    for lag in range(1, LAGS + 1):
        cross = scores[lag:].T @ scores[:-lag]
        meat += (1 - lag / (LAGS + 1)) * (cross + cross.T)
    bread = np.linalg.inv(x.T @ x)
    by_hand = np.sqrt(np.diag(bread @ meat @ bread))
    np.testing.assert_allclose([curve.std_errors[t] for t in curve.spec.terms], by_hand, rtol=1e-8)


def test_committed_dataset_has_positive_slopes_in_every_variant(committed) -> None:
    comparison = committed.comparison
    assert len(comparison) == 8
    forms = comparison.groupby("slope_form")
    assert (forms.get_group("constant")["beta"] > 0).all()
    assert (forms.get_group("proportional")["kappa"] > 0).all()
    stage = committed.base.stage_curve(de_price_profile(committed.table))
    assert (stage.slope_eur_mwh_per_gwh > 0).all()


def test_report_states_coefficients_and_fit_statistics(committed) -> None:
    report = "\n".join(committed.report)
    for name in committed.base.params:
        assert name in report
    for label in ("n =", "R2", "adjusted R2", "Durbin-Watson", "HAC", "LOYO RMSE", "Selection"):
        assert label in report


# --- Regression sample --------------------------------------------------------------------------


def test_net_import_is_per_168_hour_stage(drivers) -> None:
    cal, net_import, price_de, _ = drivers
    weekly = synthetic_weekly(net_import, price_de, np.full(len(cal), 40.0))
    table = price_table(weekly, SEASONS)
    np.testing.assert_array_equal(table["net_import_stage_gwh"], net_import)
    folded = (cal["hours"] == 2 * HOURS_PER_WEEK).to_numpy()
    assert folded.sum() == 1
    assert weekly.loc[folded, "hydro_gwh"].item() == 2 * table.loc[folded, "hydro_stage_gwh"].item()
    np.testing.assert_array_equal(table["price_de_eur_mwh"], price_de)  # prices are not rescaled
    assert table["complete"].all()


def test_incomplete_weeks_are_dropped_and_listed(drivers) -> None:
    cal, net_import, price_de, _ = drivers
    price, _ = synthetic_prices(drivers, "proportional", False, KAPPA)
    weekly = synthetic_weekly(net_import, price_de, price)
    gap = week_mask(cal, 2021, 50)
    weekly.loc[gap, "solar_gwh"] = np.nan
    table = price_table(weekly, SEASONS)
    assert not table.loc[gap, "complete"].item()
    assert table.loc[gap, "missing_inputs"].item() == "solar_gwh"
    spec = spec_of("proportional", False)
    curve = fit_curve(table, spec, LAGS)
    assert curve.nobs == len(cal) - 1
    assert curve.params == pytest.approx(fit_curve(table[~gap], spec, LAGS).params, rel=1e-12)
    result = build_price(SYNTHETIC_CFG, weekly)
    assert result.table.loc[gap, "status"].item() == "missing"
    assert "2021-W50" in "\n".join(result.report)


def test_excluded_years_leave_the_sample(drivers) -> None:
    table = synthetic_table(drivers, "constant", False, BETA)
    spec = spec_of("constant", False, exclude_years=(2020,))
    curve = fit_curve(table, spec, LAGS)
    assert curve.years == (FIRST, LAST)
    shifted = table.copy()
    shifted.loc[shifted["iso_year"] == 2020, "price_se_eur_mwh"] += 1_000.0
    assert fit_curve(shifted, spec, LAGS).params == pytest.approx(curve.params, rel=1e-12)
    with pytest.raises(PriceError, match="2032"):
        fit_curve(table, spec_of("constant", False, exclude_years=(2032,)), LAGS)


def test_loyo_does_not_see_the_left_out_year(drivers) -> None:
    table = synthetic_table(drivers, "proportional", True, KAPPA)
    spec = spec_of("proportional", True)
    loyo = loyo_predictions(table, spec, LAGS)
    assert loyo.notna().all()
    year = (table["iso_year"] == 2020).to_numpy()
    shifted = table.copy()
    shifted.loc[year, "price_se_eur_mwh"] += 5.0
    moved = loyo_predictions(shifted, spec, LAGS)
    np.testing.assert_allclose(moved[year], loyo[year], rtol=1e-10)
    assert not np.allclose(moved[~year], loyo[~year])  # the other years were fitted with 2020


# --- Weekly curve and cost function -------------------------------------------------------------


@pytest.mark.parametrize("slope_form", ["constant", "proportional"])
def test_stage_curve_reproduces_the_regression(drivers, slope_form) -> None:
    table = synthetic_table(drivers, slope_form, True, SLOPE[slope_form])
    curve = fit_curve(table, spec_of(slope_form, True), LAGS)
    rng = np.random.default_rng(5)
    weeks = np.arange(1, MODEL_WEEKS_PER_YEAR + 1)
    profile = rng.uniform(30.0, 120.0, len(weeks))
    net_import = rng.uniform(-900.0, 300.0, len(weeks))
    probe = pd.DataFrame(
        {"iso_week": weeks, "net_import_stage_gwh": net_import, "price_de_eur_mwh": profile}
    )
    stage = curve.stage_curve(profile)
    np.testing.assert_allclose(
        stage.price_eur_mwh(net_import, weeks), curve.predict_eur_mwh(probe), rtol=1e-12
    )


def flat_stage(intercept_eur_mwh: float, slope_eur_mwh_per_gwh: float) -> StageCurve:
    weeks = MODEL_WEEKS_PER_YEAR
    return StageCurve(
        np.full(weeks, intercept_eur_mwh),
        np.full(weeks, slope_eur_mwh_per_gwh),
        np.full(weeks, 60.0),
    )


def test_cost_is_the_integral_of_the_price() -> None:
    rng = np.random.default_rng(7)
    weeks = MODEL_WEEKS_PER_YEAR
    stage = StageCurve(
        rng.uniform(20.0, 80.0, weeks), rng.uniform(0.01, 0.1, weeks), rng.uniform(30, 120, weeks)
    )
    week = 10
    assert stage.cost_keur(0.0, week) == 0.0
    net_import = rng.uniform(-1_500.0, 500.0, 200)
    step = 1e-3
    derivative = (
        stage.cost_keur(net_import + step, week) - stage.cost_keur(net_import - step, week)
    ) / (2 * step)
    np.testing.assert_allclose(derivative, stage.price_eur_mwh(net_import, week), atol=1e-6)
    second = (
        stage.cost_keur(net_import + 1.0, week)
        - 2 * stage.cost_keur(net_import, week)
        + stage.cost_keur(net_import - 1.0, week)
    )
    np.testing.assert_allclose(second, stage.slope_eur_mwh_per_gwh[week - 1], rtol=1e-4)
    # Units: GWh x EUR/MWh = kEUR. 10 GWh at 50 EUR/MWh plus the slope term 0.02/2 x 10^2.
    flat = flat_stage(50.0, 0.02)
    assert flat.cost_keur(10.0, 1) == pytest.approx(501.0)
    assert flat.cost_keur(-10.0, 1) == pytest.approx(-499.0)  # exports earn revenue
    grid = rng.uniform(-1_500.0, 500.0, (5, 3, 5))  # the SDP's (K, J, K) candidates
    cost = stage.cost_keur(grid, week)
    assert cost.shape == grid.shape
    a, b = stage.intercept_eur_mwh[week - 1], stage.slope_eur_mwh_per_gwh[week - 1]
    np.testing.assert_allclose(cost, a * grid + b / 2 * grid**2)


def test_stage_curve_rejects_bad_inputs(drivers) -> None:
    weeks = MODEL_WEEKS_PER_YEAR
    with pytest.raises(PriceError, match="52"):
        StageCurve(np.full(weeks - 1, 50.0), np.full(weeks - 1, 0.02), np.full(weeks - 1, 60.0))
    with_nan = np.full(weeks, 50.0)
    with_nan[3] = np.nan
    with pytest.raises(PriceError, match="finite"):
        StageCurve(with_nan, np.full(weeks, 0.02), np.full(weeks, 60.0))
    flat_slope = np.full(weeks, 0.02)
    flat_slope[[9, 30]] = 0.0
    with pytest.raises(PriceError, match=r"10, 31"):
        StageCurve(np.full(weeks, 50.0), flat_slope, np.full(weeks, 60.0))
    table = synthetic_table(drivers, "proportional", False, KAPPA)
    curve = fit_curve(table, spec_of("proportional", False), LAGS)
    profile = np.full(weeks, 60.0)
    profile[5] = -5.0  # a negative German price makes a proportional slope negative
    with pytest.raises(PriceError, match="6"):
        curve.stage_curve(profile)
    stage = flat_stage(50.0, 0.02)
    with pytest.raises(ValueError, match="week"):
        stage.price_eur_mwh(0.0, 0)
    with pytest.raises(ValueError, match="week"):
        stage.cost_keur(0.0, weeks + 1)


def test_de_price_profile_and_levels(drivers) -> None:
    cal, net_import, _, _ = drivers
    # Mean German price over the three years: ISO week + 10 EUR/MWh
    price_de = (cal["iso_week"] + 10.0 * (cal["iso_year"] - FIRST)).to_numpy(dtype=float)
    table = price_table(synthetic_weekly(net_import, price_de, price_de), SEASONS)
    table.loc[0, "price_de_eur_mwh"] = np.nan  # 2019-W01: the mean of 2020 and 2021 remains
    profile = de_price_profile(table)
    expected = np.arange(1, MODEL_WEEKS_PER_YEAR + 1) + 10.0
    expected[0] = 1.0 + 15.0
    np.testing.assert_allclose(profile, expected)
    np.testing.assert_allclose(
        de_price_profile(table, years=[LAST]), np.arange(1, MODEL_WEEKS_PER_YEAR + 1) + 20.0
    )
    scaled = de_price_profile(table, level_eur_mwh=80.0)
    assert scaled.mean() == pytest.approx(80.0)
    np.testing.assert_allclose(scaled / profile, 80.0 / profile.mean())


def test_loyo_accuracy_on_hand_made_errors() -> None:
    errors = pd.Series([1.0, -1.0, 3.0, -3.0, 2.0, np.nan])
    season = pd.Series(["Q1", "Q1", "Q2", "Q2", "Q3", "Q3"])
    before = pd.Series([True, True, True, False, False, False])
    scores = loyo_accuracy(errors, season, {"before": before, "after": ~before})
    assert scores["loyo_rmse_eur_mwh"] == pytest.approx(np.sqrt(24 / 5))  # the gap is skipped
    assert scores["loyo_rmse_before_eur_mwh"] == pytest.approx(np.sqrt(11 / 3))
    assert scores["loyo_rmse_after_eur_mwh"] == pytest.approx(np.sqrt(13 / 2))
    assert scores["loyo_season_bias_eur_mwh"] == pytest.approx(2.0)  # Q3; Q1 and Q2 average 0


def test_variants_must_be_scored_on_the_same_weeks(drivers) -> None:
    table = synthetic_table(drivers, "proportional", False, KAPPA)
    variants = {}
    for slope_form in ("constant", "proportional"):
        spec = spec_of(slope_form, False)
        variants[spec.label] = Variant(
            fit_curve(table, spec, LAGS), loyo_predictions(table, spec, LAGS)
        )
    assert len(compare_variants(table, variants, [LAST])) == 2
    label = list(variants)[1]
    loyo = variants[label].loyo_eur_mwh.copy()
    loyo.iloc[:10] = np.nan
    variants[label] = Variant(variants[label].curve, loyo)
    with pytest.raises(PriceError, match="other weeks"):
        compare_variants(table, variants, [LAST])


def test_selection_rule() -> None:
    comparison = pd.DataFrame(
        {
            "loyo_rmse_eur_mwh": [17.0, 16.7, 16.9, 21.5],
            "loyo_season_bias_eur_mwh": [1.0, 4.4, 1.8, 0.5],
            "n_params": [4, 3, 6, 3],
        },
        index=["a", "b", "c", "d"],
    )
    assert select_variant(comparison, 0.0) == "b"  # no tolerance: the lowest RMSE
    assert select_variant(comparison, 0.5) == "a"  # a, b and c tie: the smallest seasonal bias
    comparison.loc["c", "loyo_season_bias_eur_mwh"] = 1.0
    assert select_variant(comparison, 0.5) == "a"  # a and c tie again: fewer parameters


def test_curve_json_round_trip(tmp_path, drivers) -> None:
    table = synthetic_table(drivers, "proportional", True, KAPPA)
    curve = fit_curve(table, spec_of("proportional", True), LAGS)
    path = tmp_path / "price_curve.json"
    save_curve(curve, path)
    loaded = load_curve(path)
    assert loaded == curve
    profile = np.linspace(30.0, 120.0, MODEL_WEEKS_PER_YEAR)
    np.testing.assert_array_equal(
        loaded.stage_curve(profile).slope_eur_mwh_per_gwh,
        curve.stage_curve(profile).slope_eur_mwh_per_gwh,
    )


def test_load_curve_rejects_a_broken_file(tmp_path, drivers) -> None:
    table = synthetic_table(drivers, "proportional", False, KAPPA)
    good = fit_curve(table, spec_of("proportional", False), LAGS).to_dict()
    without_gamma = {name: value for name, value in good["params"].items() if name != "gamma"}
    broken = {
        "slope": {**good, "params": {**good["params"], "kappa": -1e-4}},
        "coefficients": {**good, "params": without_gamma},
        "standard errors": {**good, "std_errors": {**good["std_errors"], "beta": 1.0}},
        "units": {**good, "units": {**good["units"], "kappa": "EUR/MWh"}},
    }
    path = tmp_path / "price_curve.json"
    for message, raw in broken.items():
        path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(PriceError, match=message):
            load_curve(path)


@pytest.mark.parametrize("slope_form", ["constant", "proportional"])
def test_de_pass_through_is_the_derivative_in_the_german_price(drivers, slope_form) -> None:
    table = synthetic_table(drivers, slope_form, True, SLOPE[slope_form])
    curve = fit_curve(table, spec_of(slope_form, True), LAGS)
    probe = table.head(20)
    shifted = probe.assign(price_de_eur_mwh=probe["price_de_eur_mwh"] + 1.0)
    derivative = curve.predict_eur_mwh(shifted) - curve.predict_eur_mwh(probe)  # linear in p_DE
    np.testing.assert_allclose(
        curve.de_pass_through(probe["net_import_stage_gwh"]), derivative, rtol=1e-9
    )
    assert (curve.kappa_per_gwh is None) == (slope_form == "constant")


# --- Committed dataset and CLI ------------------------------------------------------------------


def test_committed_dataset_drops_only_documented_gaps(committed) -> None:
    gaps = load_known_gaps(PROJECT_ROOT / CFG.data.known_gaps_file)
    inputs = set(REGRESSION_INPUTS)
    expected = sorted(w for gap in gaps if inputs & set(gap["columns"]) for w in gap["weeks"])
    table = committed.table
    dropped = table[table["status"] == "missing"]
    labels = sorted(
        week_label(y, k) for y, k in zip(dropped["iso_year"], dropped["iso_week"], strict=True)
    )
    assert labels == expected
    assert (table["status"] == "used").sum() == committed.base.nobs
    assert committed.comparison["configured_base"].sum() == 1
    assert committed.comparison["selected"].sum() == 1


def test_configured_base_model_follows_the_selection_rule(committed) -> None:
    # If new data makes the rule pick another variant, config must be changed deliberately.
    assert committed.selected == committed.base.spec.label


def _config_with(tmp_path, **sections: dict) -> str:
    """Write a copy of the base config with some section values replaced; return its path."""
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    for section, values in sections.items():
        raw[section].update(values)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(path)


def test_cli_price_writes_the_outputs(tmp_path, committed) -> None:
    outputs = {
        "weekly_csv": str(tmp_path / "price_weekly.csv"),
        "variants_csv": str(tmp_path / "price_variants.csv"),
        "curve_json": str(tmp_path / "price_curve.json"),
        "report": str(tmp_path / "price_report.txt"),
    }
    config = _config_with(tmp_path, price_model=outputs)
    result = CliRunner().invoke(app, ["price", "--config", config])
    assert result.exit_code == 0, result.output
    assert load_curve(Path(outputs["curve_json"])) == committed.base
    variants = pd.read_csv(outputs["variants_csv"])
    assert len(variants) == 8
    assert variants["configured_base"].sum() == 1
    weekly = pd.read_csv(outputs["weekly_csv"])
    assert {"net_import_stage_gwh", "fitted_eur_mwh", "loyo_eur_mwh", "status"} <= set(
        weekly.columns
    )
    assert "LOYO RMSE" in Path(outputs["report"]).read_text(encoding="utf-8")


def test_cli_price_stops_on_a_negative_slope(tmp_path, drivers) -> None:
    _, net_import, price_de, _ = drivers
    price, _ = synthetic_prices(drivers, "constant", False, -BETA)
    csv = tmp_path / "weekly.csv"
    write_weekly_csv(synthetic_weekly(net_import, price_de, price), csv, CFG.data.csv_float_format)
    config = _config_with(
        tmp_path,
        period={"first_iso_year": FIRST, "last_iso_year": LAST},
        data={"weekly_csv": str(csv)},
        price_model={
            "crisis_years": [LAST],
            "exclude_years": [],
            "report": str(tmp_path / "report.txt"),
        },
    )
    result = CliRunner().invoke(app, ["price", "--config", config])
    assert result.exit_code == 1
    assert "slope" in result.output
    assert not (tmp_path / "report.txt").exists()
