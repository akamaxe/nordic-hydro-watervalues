"""All figures, with one consistent and accessible style (static matplotlib, renders on GitHub).

Colours follow a validated categorical palette in a fixed order (never cycled past eight); context
series are drawn in a recessive grey, and every figure keeps a single y-axis.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler
from matplotlib.axes import Axes

from watervalues.data.aggregate import per_stage

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CONTEXT = "#c3c2b7"  # recessive lines that give context (e.g. other years)
SERIES = (  # categorical slots, in this fixed order
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
)
LINE_WIDTH = 1.6
CONTEXT_WIDTH = 0.9


def use_style() -> None:
    """Apply the project style to matplotlib."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titlelocation": "left",
            "axes.titlecolor": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "lines.linewidth": LINE_WIDTH,
            "legend.frameon": False,
            "legend.labelcolor": INK_SECONDARY,
            "axes.prop_cycle": cycler(color=list(SERIES)),
        }
    )


def _axes(ax: Axes | None, size: tuple[float, float] = (9, 4)) -> Axes:
    if ax is not None:
        return ax
    _, new_ax = plt.subplots(figsize=size)
    return new_ax


def _label_end(ax: Axes, x: object, y: float, text: str, color: str) -> None:
    """Direct label at the end of a line (text in ink, a coloured mark carries identity)."""
    ax.plot([x], [y], marker="o", markersize=4, color=color, zorder=4)
    ax.annotate(
        text,
        (x, y),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        color=INK_SECONDARY,
        fontsize=9,
    )


def reservoir_filling(
    weekly: pd.DataFrame,
    capacity_gwh: float,
    highlight: Sequence[int] = (),
    ax: Axes | None = None,
) -> Axes:
    """Storage at the start of each week [% of capacity], one line per ISO year.

    Parameters
    ----------
    weekly : pandas.DataFrame
        Weekly dataset with ``iso_year``, ``iso_week`` and ``storage_gwh`` [GWh].
    capacity_gwh : float
        Storage capacity used as 100 % [GWh].
    highlight : sequence of int
        Years drawn in colour with a direct label (at most eight); the rest are context lines.
    ax : matplotlib Axes or None
        Axes to draw on; a new figure is created when None.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if len(highlight) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} highlighted years")
    ax = _axes(ax)
    colours = dict(zip(highlight, SERIES, strict=False))
    context_drawn = False
    for year, group in weekly.groupby("iso_year"):
        filling = 100 * group["storage_gwh"] / capacity_gwh
        if year in colours:
            continue
        ax.plot(
            group["iso_week"],
            filling,
            color=CONTEXT,
            linewidth=CONTEXT_WIDTH,
            label=None if context_drawn else "other years",
            zorder=2,
        )
        context_drawn = True
    for year in highlight:
        group = weekly[weekly["iso_year"] == year]
        filling = 100 * group["storage_gwh"] / capacity_gwh
        ax.plot(group["iso_week"], filling, color=colours[year], label=str(year), zorder=3)
        valid = filling.dropna()
        if not valid.empty:
            last = valid.index[-1]
            _label_end(ax, group.loc[last, "iso_week"], valid.loc[last], str(year), colours[year])
    ax.set_xlim(1, 54)
    ax.set_ylim(0, 100)
    ax.set_xlabel("ISO week")
    ax.set_ylabel(f"Storage at start of week [% of {capacity_gwh / 1e3:.1f} TWh]")
    ax.legend(loc="lower left", ncols=len(highlight) + 1)
    return ax


def hydro_and_load(weekly: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Weekly hydro production and total load [GWh per 168-hour week].

    Folded 53-week years are shown per standard week (see ``per_stage``).
    """
    ax = _axes(ax)
    stage = per_stage(weekly, ["hydro_gwh", "load_gwh"])
    x = pd.DatetimeIndex(weekly["week_start"])
    for column, name, colour in (
        ("hydro_gwh", "hydro production", SERIES[0]),
        ("load_gwh", "total load", SERIES[1]),
    ):
        ax.plot(x, stage[column], color=colour, label=name)
    ax.set_ylabel("Energy [GWh per week]")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower left", ncols=2)
    return ax


def prices_se_de(weekly: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Weekly day-ahead prices: Sweden (load-weighted SE1-SE4) and Germany [EUR/MWh]."""
    ax = _axes(ax)
    x = pd.DatetimeIndex(weekly["week_start"])
    ax.plot(x, weekly["price_se_eur_mwh"], color=SERIES[0], label="Sweden (load-weighted)")
    ax.plot(x, weekly["price_de_eur_mwh"], color=SERIES[1], label="Germany (DE-AT-LU, DE-LU)")
    ax.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
    ax.set_ylabel("Day-ahead price [EUR/MWh]")
    ax.legend(loc="upper left", ncols=2)
    return ax
