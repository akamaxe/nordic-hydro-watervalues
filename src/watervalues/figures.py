"""The figures of the README and ``reports/figures/``, drawn from the model outputs.

``watervalues figures`` reads what ``watervalues solve`` and ``watervalues simulate`` wrote to
``outputs/`` (and recomputes the price curve, which takes a second) and saves static PNGs, so
that the README shows exactly the results of the last run.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from watervalues import plots
from watervalues.config import PROJECT_ROOT, AppConfig
from watervalues.data.build import load_weekly
from watervalues.price import build_price
from watervalues.sdp import load_solution


class FiguresError(RuntimeError):
    """An output that a figure needs is missing."""


def _require(path: Path, command: str) -> Path:
    if not path.exists():
        raise FiguresError(f"{path} is missing: run `watervalues {command}` first")
    return path


def make_figures(cfg: AppConfig) -> list[Path]:
    """Draw every figure and save it as PNG in ``figures.dir``.

    Parameters
    ----------
    cfg : AppConfig
        Configuration: the output paths of ``sdp`` and ``simulation`` and ``figures.dir``.

    Returns
    -------
    list of Path
        The files written.

    Raises
    ------
    FiguresError
        If the outputs of ``watervalues solve`` or ``watervalues simulate`` are missing.
    """
    sdp, sim = cfg.sdp, cfg.simulation
    solution = load_solution(_require(PROJECT_ROOT / sdp.values_npz, "solve"))
    backtest = pd.read_csv(
        _require(PROJECT_ROOT / sim.backtest_csv, "simulate"), parse_dates=["week_start"]
    )
    de_price = pd.read_csv(
        _require(PROJECT_ROOT / sim.de_price_csv, "simulate"), index_col="de_price_level_eur_mwh"
    )
    dry_wet = pd.read_csv(_require(PROJECT_ROOT / sim.dry_wet_csv, "simulate"))
    weekly = load_weekly(PROJECT_ROOT / cfg.data.weekly_csv)
    price = build_price(cfg, weekly)
    capacity = solution.problem.capacity_gwh
    plots.use_style()

    def heatmap() -> Figure:
        ax = plots.water_value_heatmap(
            solution.water_value_eur_mwh, solution.problem.storage_gwh, weekly
        )
        ax.set_title("Water value of the Swedish reservoir, with the observed storage 2015-2025")
        return ax.figure

    def slopes() -> Figure:
        ax = plots.slope_by_year(
            price.slopes_by_year,
            cfg.price_model.crisis_years,
            kappa_per_gwh=price.base.kappa_per_gwh,
        )
        ax.set_title("Price slope of each year against its German price level")
        return ax.figure

    def storage() -> Figure:
        fig = plots.backtest_storage(backtest, capacity)
        fig.suptitle("Backtest: simulated and observed storage", x=0.01, ha="left")
        return fig

    def prices() -> Figure:
        ax = plots.backtest_prices(backtest)
        ax.set_title("Backtest: simulated and observed weekly price")
        return ax.figure

    def continental() -> Figure:
        ax = plots.de_price_scenario(de_price)
        ax.set_title("Water value against the German price level")
        return ax.figure

    def dry_and_wet() -> Figure:
        fig = plots.dry_and_wet(dry_wet, capacity)
        fig.suptitle("Driest and wettest year from the same start", x=0.01, ha="left")
        return fig

    drawings: dict[str, Callable[[], Figure]] = {
        "water_values.png": heatmap,
        "price_slope_by_year.png": slopes,
        "backtest_storage.png": storage,
        "backtest_prices.png": prices,
        "scenario_de_price.png": continental,
        "scenario_dry_wet.png": dry_and_wet,
    }
    folder = PROJECT_ROOT / cfg.figures.dir
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for name, draw in drawings.items():
        fig = draw()
        path = folder / name
        fig.savefig(path, metadata={"Software": None})
        plt.close(fig)
        written.append(path)
    return written
