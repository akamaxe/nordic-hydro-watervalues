"""Smoke tests for the figures (non-interactive backend)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from watervalues import plots
from watervalues.data.calendar import week_calendar


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
