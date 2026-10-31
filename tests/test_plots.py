"""Smoke tests for the figures (non-interactive backend)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from watervalues import plots
from watervalues.data.calendar import week_calendar
from watervalues.price import StageCurve


@pytest.fixture
def weekly() -> pd.DataFrame:
    cal = week_calendar(2019, 2021)
    n = len(cal)
    rng = np.random.default_rng(0)
    return cal.assign(
        storage_gwh=20_000 + 5_000 * np.sin(np.arange(n) / 8),
        hydro_gwh=1500.0 * cal["hours"] / 168,
        load_gwh=2600.0 * cal["hours"] / 168,
        price_se_eur_mwh=40 + rng.normal(0, 5, n),
        price_de_eur_mwh=60 + rng.normal(0, 5, n),
    )


def test_reservoir_filling_highlights_years(weekly) -> None:
    plots.use_style()
    ax = plots.reservoir_filling(weekly, 33_700, highlight=[2020, 2021])
    colours = {line.get_label(): line.get_color() for line in ax.get_lines()}
    assert colours["2020"] == plots.SERIES[0]
    assert colours["other years"] == plots.CONTEXT
    plt.close("all")


def test_too_many_highlights_are_rejected(weekly) -> None:
    with pytest.raises(ValueError, match="at most"):
        plots.reservoir_filling(weekly, 33_700, highlight=list(range(9)))


def test_time_series_figures_use_one_axis(weekly) -> None:
    for draw in (plots.hydro_and_load, plots.prices_se_de):
        ax = draw(weekly)
        assert len(ax.figure.axes) == 1  # never a second y-axis
        assert ax.get_legend() is not None
    folded = plots.hydro_and_load(weekly).get_lines()[0].get_ydata()
    np.testing.assert_allclose(folded, 1500.0)  # the 336-hour week is shown per 168 hours
    plt.close("all")


@pytest.fixture
def scenarios() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    weeks = range(1, 53)
    return pd.DataFrame(
        rng.uniform(300, 4_000, (3, len(weeks))), index=[2019, 2020, 2021], columns=weeks
    )


def test_inflow_climatology_highlights_years(scenarios) -> None:
    ax = plots.inflow_climatology(scenarios, highlight=[2020, 2019])
    colours = {line.get_label(): line.get_color() for line in ax.get_lines()}
    assert colours["2020"] == plots.SERIES[0]
    assert colours["2019"] == plots.SERIES[1]
    assert len(ax.figure.axes) == 1
    mean = next(line for line in ax.get_lines() if line.get_label() == "mean")
    np.testing.assert_allclose(mean.get_ydata(), scenarios.mean(axis=0))
    plt.close("all")


def test_annual_inflow_colours_the_dry_and_wet_years() -> None:
    annual = pd.DataFrame(
        # 2019: production unknown (NaN), so no marker is drawn
        {"inflow_model_twh": [60.0, 75.0, 55.0], "hydro_model_twh": [np.nan, 68.0, 62.0]},
        index=pd.Index([2019, 2020, 2021], name="iso_year"),
    )
    ax = plots.annual_inflow(annual, dry=2021, wet=2020)
    bars = {round(p.get_x() + p.get_width() / 2): p for p in ax.patches}
    assert bars[2020].get_facecolor() == matplotlib.colors.to_rgba(plots.SERIES[0])
    assert bars[2021].get_facecolor() == matplotlib.colors.to_rgba(plots.SERIES[1])
    assert bars[2019].get_facecolor() == matplotlib.colors.to_rgba(plots.CONTEXT)
    assert len(ax.figure.axes) == 1
    plt.close("all")


@pytest.fixture
def price_frame() -> pd.DataFrame:
    cal = week_calendar(2019, 2021)
    n = len(cal)
    rng = np.random.default_rng(1)
    price = 40 + rng.normal(0, 5, n)
    fitted = price + rng.normal(0, 2, n)
    loyo = price + rng.normal(0, 3, n)
    return cal.assign(
        net_import_stage_gwh=rng.uniform(-900, 300, n),
        price_se_eur_mwh=price,
        fitted_eur_mwh=fitted,
        loyo_eur_mwh=loyo,
        residual_eur_mwh=price - fitted,
        loyo_residual_eur_mwh=price - loyo,
        season=np.where(cal["iso_week"] < 27, "Q1", "Q3"),
        complete=True,
    )


def test_price_vs_net_import_colours_the_periods(price_frame) -> None:
    ax = plots.price_vs_net_import(price_frame, crisis_years=[2020])
    colours = {c.get_label(): c.get_facecolor()[0] for c in ax.collections}
    assert list(colours) == ["2019", "2020", "2021"]
    np.testing.assert_allclose(colours["2019"][:3], matplotlib.colors.to_rgb(plots.SERIES[0]))
    assert len(ax.figure.axes) == 1
    with pytest.raises(ValueError, match="crisis_years"):
        plots.price_vs_net_import(price_frame, crisis_years=[])
    plt.close("all")


def test_slope_by_year_draws_the_proportional_line() -> None:
    slopes = pd.DataFrame(
        {
            "beta_eur_mwh_per_gwh": [0.02, 0.03, 0.2],
            "beta_se_eur_mwh_per_gwh": [0.005, 0.004, 0.03],
            "price_de_mean_eur_mwh": [30.0, 40.0, 240.0],
        },
        index=pd.Index([2019, 2020, 2022], name="iso_year"),
    )
    ax = plots.slope_by_year(slopes, crisis_years=[2020], kappa_per_gwh=0.0009)
    line = next(line for line in ax.get_lines() if "proportional" in line.get_label())
    np.testing.assert_allclose(line.get_ydata(), 0.0009 * line.get_xdata())
    # Labelled one by one from the first crisis year on; the earlier years are in the legend.
    assert [text.get_text() for text in ax.texts] == ["2020", "2022"]
    legend = [text.get_text() for text in ax.get_legend().get_texts()]
    assert any(label.startswith("2019,") for label in legend)
    assert len(ax.figure.axes) == 1
    plt.close("all")


def test_price_fit_and_residual_figures(price_frame) -> None:
    for draw in (plots.price_fit, plots.residuals_over_time):
        ax = draw(price_frame)
        assert len(ax.figure.axes) == 1
        assert ax.get_legend() is not None
    ax = plots.residuals_by_season(price_frame, ["Q1", "Q3"])
    assert [label.get_text() for label in ax.get_xticklabels()] == ["Q1", "Q3"]
    assert len(ax.patches) == 4  # an in-sample and a year-left-out box per season
    assert len(ax.figure.axes) == 1
    plt.close("all")


def test_stage_slopes_draw_one_line_per_profile() -> None:
    weeks = 52
    curves = {
        "mean": StageCurve(
            np.full(weeks, 50.0), np.linspace(0.04, 0.1, weeks), np.full(weeks, 70.0)
        ),
        "120 EUR/MWh": StageCurve(
            np.full(weeks, 90.0), np.linspace(0.06, 0.16, weeks), np.full(weeks, 120.0)
        ),
    }
    ax = plots.stage_slopes(curves)
    lines = {line.get_label(): line for line in ax.get_lines()}
    assert lines["mean"].get_color() == plots.SERIES[0]
    np.testing.assert_allclose(
        lines["120 EUR/MWh"].get_ydata(), curves["120 EUR/MWh"].slope_eur_mwh_per_gwh
    )
    assert len(ax.figure.axes) == 1
    plt.close("all")
