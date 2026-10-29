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
