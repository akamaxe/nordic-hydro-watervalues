"""All figures, with one consistent and accessible style (static matplotlib, renders on GitHub).

Colours follow a validated categorical palette in a fixed order (never cycled past eight); context
series are drawn in a recessive grey, and every figure keeps a single y-axis.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from watervalues.data.aggregate import per_stage

if TYPE_CHECKING:
    from watervalues.price import StageCurve

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
MARKER_AREA = 14  # scatter markers [pt^2]
BOX_WIDTH = 0.35  # width of each box in a pair, in units of the category spacing
BACKTEST_COLUMNS = 4  # panels per row in the backtest figure
HEATMAP_CAP_PERCENTILE = 99  # the heatmap's colour scale ends here (empty storage is extreme)


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


def inflow_climatology(
    scenarios: pd.DataFrame, highlight: Sequence[int] = (), ax: Axes | None = None
) -> Axes:
    """Weekly inflow per stage: range and mean over the years, plus highlighted years.

    Parameters
    ----------
    scenarios : pandas.DataFrame
        Scenario matrix (ISO years x stages 1-52) [GWh per 168-hour stage].
    highlight : sequence of int
        Years drawn in colour, in the fixed palette order (at most eight).
    ax : matplotlib Axes or None
        Axes to draw on; a new figure is created when None.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if len(highlight) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} highlighted years")
    ax = _axes(ax)
    weeks = scenarios.columns.to_numpy()
    ax.fill_between(
        weeks,
        scenarios.min(axis=0),
        scenarios.max(axis=0),
        color=CONTEXT,
        alpha=0.6,
        linewidth=0,
        label="range over the years",
        zorder=1,
    )
    ax.plot(weeks, scenarios.mean(axis=0), color=INK_SECONDARY, linestyle="--", label="mean")
    for year, colour in zip(highlight, SERIES, strict=False):
        ax.plot(weeks, scenarios.loc[year], color=colour, label=str(year), zorder=3)
    ax.set_xlim(1, weeks.max())
    ax.set_ylim(bottom=0)
    ax.set_xlabel("ISO week (model stage)")
    ax.set_ylabel("Inflow [GWh per week]")
    ax.legend(loc="upper right")
    return ax


def annual_inflow(annual: pd.DataFrame, dry: int, wet: int, ax: Axes | None = None) -> Axes:
    """Annual inflow (bars) and hydro production (markers) over 52 standard weeks [TWh].

    Parameters
    ----------
    annual : pandas.DataFrame
        Indexed by ISO year with ``inflow_model_twh`` and ``hydro_model_twh`` [TWh]
        (see :func:`watervalues.inflow.annual_totals`).
    dry, wet : int
        Years drawn in colour (the wet one in the first palette slot, the dry one in the second).
    ax : matplotlib Axes or None
        Axes to draw on; a new figure is created when None.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _axes(ax)
    years = annual.index.to_numpy()
    colours = {wet: SERIES[0], dry: SERIES[1]}
    ax.bar(
        years,
        annual["inflow_model_twh"],
        color=[colours.get(year, CONTEXT) for year in years],
        width=0.7,
        zorder=2,
    )
    ax.plot(
        years,
        annual["hydro_model_twh"],
        linestyle="none",
        marker="o",
        markersize=5,
        color=INK,
        zorder=3,
    )
    handles = [
        Patch(color=SERIES[0], label=f"inflow, wettest year ({wet})"),
        Patch(color=SERIES[1], label=f"inflow, driest year ({dry})"),
        Patch(color=CONTEXT, label="inflow, other years"),
        Line2D([], [], linestyle="none", marker="o", color=INK, label="hydro production"),
    ]
    ax.set_xticks(years)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("ISO year (52 standard weeks)")
    ax.set_ylabel("Energy [TWh per year]")
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncols=2)
    return ax


def _span(first: int, last: int) -> str:
    return str(first) if first == last else f"{first}-{last}"


def price_vs_net_import(
    table: pd.DataFrame, crisis_years: Sequence[int], ax: Axes | None = None
) -> Axes:
    """Weekly Swedish price against net import, coloured by period.

    The periods are the years before, during and after ``crisis_years``.

    Parameters
    ----------
    table : pandas.DataFrame
        Price table (see :func:`watervalues.price.price_table`) with ``iso_year``,
        ``net_import_stage_gwh`` [GWh per 168-hour stage], ``price_se_eur_mwh`` [EUR/MWh] and
        ``complete``; incomplete weeks are not drawn.
    crisis_years : sequence of int
        Years of the middle period (at least one).
    ax : matplotlib Axes or None
        Axes to draw on; a new figure is created when None.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if not crisis_years:
        raise ValueError("crisis_years must name at least one year")
    ax = _axes(ax, size=(7, 4.5))
    rows = table[table["complete"]]
    years = rows["iso_year"]
    first, last = min(crisis_years), max(crisis_years)
    periods = (
        (years < first, _span(int(years.min()), first - 1)),
        (years.isin(list(crisis_years)), _span(first, last)),
        (years > last, _span(last + 1, int(years.max()))),
    )
    for (mask, label), colour in zip(periods, SERIES, strict=False):
        if mask.any():
            ax.scatter(
                rows.loc[mask, "net_import_stage_gwh"],
                rows.loc[mask, "price_se_eur_mwh"],
                s=MARKER_AREA,
                color=colour,
                alpha=0.75,
                linewidths=0,
                label=label,
                zorder=2,
            )
    ax.axvline(0, color=AXIS, linewidth=0.8, zorder=1)
    ax.set_xlabel("Net import [GWh per week] (negative: Sweden exports)")
    ax.set_ylabel("Swedish price [EUR/MWh]")
    ax.legend(loc="upper left")
    return ax


def slope_by_year(
    slopes: pd.DataFrame,
    crisis_years: Sequence[int],
    kappa_per_gwh: float | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Price slope of each year fitted alone against the year's mean German price.

    Points are coloured by period (before, during and after ``crisis_years``), with error bars
    of ±2 HAC standard errors. The years from the crisis on are labelled one by one; the
    earlier years crowd at low prices and are named by the legend. With ``kappa_per_gwh``, the
    slope of the proportional form, ``kappa * p_DE``, is drawn as a line through the origin.

    Parameters
    ----------
    slopes : pandas.DataFrame
        Output of :func:`watervalues.price.slope_by_year`, indexed by ISO year, with
        ``beta_eur_mwh_per_gwh``, ``beta_se_eur_mwh_per_gwh`` [EUR/MWh per GWh] and
        ``price_de_mean_eur_mwh`` [EUR/MWh].
    crisis_years : sequence of int
        Years of the middle period (at least one).
    kappa_per_gwh : float or None
        Slope per unit of German price of the proportional form [1/GWh].
    ax : matplotlib Axes or None
        Axes to draw on; a new figure is created when None.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if not crisis_years:
        raise ValueError("crisis_years must name at least one year")
    ax = _axes(ax, size=(7, 4.5))
    years = slopes.index.to_series()
    first, last = min(crisis_years), max(crisis_years)
    periods = (
        (years < first, _span(int(years.min()), first - 1)),
        (years.isin(list(crisis_years)), _span(first, last)),
        (years > last, _span(last + 1, int(years.max()))),
    )
    for (mask, label), colour in zip(periods, SERIES, strict=False):
        rows = slopes[mask.to_numpy()]
        if rows.empty:
            continue
        ax.errorbar(
            rows["price_de_mean_eur_mwh"],
            rows["beta_eur_mwh_per_gwh"],
            yerr=2 * rows["beta_se_eur_mwh_per_gwh"],
            fmt="o",
            markersize=5,
            color=colour,
            elinewidth=1.0,
            capsize=0,
            label=f"{label}, each year fitted alone (±2 HAC SE)",
            zorder=3,
        )
    for year, row in slopes[(years >= first).to_numpy()].iterrows():
        ax.annotate(
            str(year),
            (row["price_de_mean_eur_mwh"], row["beta_eur_mwh_per_gwh"]),
            xytext=(5, 4),
            textcoords="offset points",
            color=INK_SECONDARY,
            fontsize=8,
        )
    if kappa_per_gwh is not None:
        grid = np.linspace(0.0, slopes["price_de_mean_eur_mwh"].max(), 50)
        ax.plot(
            grid,
            kappa_per_gwh * grid,
            color=SERIES[1],
            label=r"proportional slope $\kappa\, p_{DE}$",
            zorder=2,
        )
    ax.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
    ax.set_xlim(left=0)
    ax.set_xlabel("Mean German price of the year [EUR/MWh]")
    ax.set_ylabel("Price slope [EUR/MWh per GWh/week]")
    ax.legend(loc="upper left")
    return ax


def price_fit(table: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Observed weekly price, the model's in-sample fit and its leave-one-year-out prediction.

    Parameters
    ----------
    table : pandas.DataFrame
        With ``week_start``, ``price_se_eur_mwh``, ``fitted_eur_mwh`` and ``loyo_eur_mwh``
        [EUR/MWh] (the table of :class:`watervalues.price.PriceResult`).
    ax : matplotlib Axes or None
        Axes to draw on; a new figure is created when None.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _axes(ax)
    x = pd.DatetimeIndex(table["week_start"])
    ax.plot(x, table["price_se_eur_mwh"], color=SERIES[0], label="observed")
    ax.plot(x, table["fitted_eur_mwh"], color=SERIES[1], linewidth=1.1, label="model, in-sample")
    ax.plot(
        x,
        table["loyo_eur_mwh"],
        color=SERIES[2],
        linewidth=1.0,
        linestyle="--",
        label="model, year left out",
    )
    ax.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
    ax.set_ylabel("Swedish price [EUR/MWh]")
    ax.legend(loc="upper left", ncols=3)
    return ax


def residuals_over_time(table: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Residuals of the model over time, in-sample and with the year left out [EUR/MWh].

    Parameters
    ----------
    table : pandas.DataFrame
        With ``week_start``, ``residual_eur_mwh`` and ``loyo_residual_eur_mwh`` (observed minus
        model) [EUR/MWh].
    ax : matplotlib Axes or None
        Axes to draw on; a new figure is created when None.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _axes(ax)
    x = pd.DatetimeIndex(table["week_start"])
    ax.plot(
        x,
        table["loyo_residual_eur_mwh"],
        color=CONTEXT,
        linewidth=CONTEXT_WIDTH,
        label="year left out",
        zorder=2,
    )
    ax.plot(x, table["residual_eur_mwh"], color=SERIES[0], linewidth=1.0, label="in-sample")
    ax.axhline(0, color=INK_SECONDARY, linewidth=0.8, zorder=1)
    ax.set_ylabel("Residual, observed - model [EUR/MWh]")
    ax.legend(loc="upper left", ncols=2)
    return ax


def residuals_by_season(
    table: pd.DataFrame, seasons: Sequence[str], ax: Axes | None = None
) -> Axes:
    """Residuals of the model by season: in-sample and leave-one-year-out boxes [EUR/MWh].

    Parameters
    ----------
    table : pandas.DataFrame
        With ``season``, ``residual_eur_mwh`` and ``loyo_residual_eur_mwh`` [EUR/MWh].
    seasons : sequence of str
        Season names, in the order to draw.
    ax : matplotlib Axes or None
        Axes to draw on; a new figure is created when None.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _axes(ax, size=(7, 4.5))
    names = list(seasons)
    positions = np.arange(len(names), dtype=float)
    handles = []
    for offset, column, colour, label in (
        (-BOX_WIDTH / 2, "residual_eur_mwh", SERIES[0], "in-sample"),
        (BOX_WIDTH / 2, "loyo_residual_eur_mwh", SERIES[1], "year left out"),
    ):
        data = [table.loc[table["season"] == name, column].dropna().to_numpy() for name in names]
        boxes = ax.boxplot(
            data,
            positions=positions + offset,
            widths=0.9 * BOX_WIDTH,
            patch_artist=True,
            medianprops={"color": INK},
            whiskerprops={"color": INK_SECONDARY},
            capprops={"color": INK_SECONDARY},
            flierprops={
                "marker": "o",
                "markersize": 3,
                "markerfacecolor": colour,
                "markeredgewidth": 0,
                "alpha": 0.6,
            },
        )
        for patch in boxes["boxes"]:
            patch.set_facecolor(colour)
            patch.set_edgecolor(INK_SECONDARY)
            patch.set_alpha(0.8)
        handles.append(Patch(color=colour, label=label))
    ax.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
    ax.set_xticks(positions, names)
    ax.set_xlabel("Season")
    ax.set_ylabel("Residual, observed - model [EUR/MWh]")
    ax.legend(handles=handles, loc="upper left")
    return ax


def stage_slopes(curves: Mapping[str, StageCurve], ax: Axes | None = None) -> Axes:
    """Weekly slope ``b_t`` of the price curve, one line per German price profile.

    Parameters
    ----------
    curves : mapping of str to StageCurve
        Label to weekly curve (at most eight), e.g. the average profile and the scenario levels.
    ax : matplotlib Axes or None
        Axes to draw on; a new figure is created when None.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if len(curves) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} curves")
    ax = _axes(ax)
    for (label, curve), colour in zip(curves.items(), SERIES, strict=False):
        slope = curve.slope_eur_mwh_per_gwh
        ax.plot(np.arange(1, len(slope) + 1), slope, color=colour, label=label)
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("ISO week (model stage)")
    ax.set_ylabel("Price slope b_t [EUR/MWh per GWh/week]")
    ax.legend(loc="upper left", ncols=len(curves))
    return ax


def water_value_heatmap(
    water_value_eur_mwh: np.ndarray,
    storage_gwh: np.ndarray,
    weekly: pd.DataFrame | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Water value by ISO week and storage level, with the historical storage paths on top.

    Parameters
    ----------
    water_value_eur_mwh : numpy.ndarray
        Water values ``(52, K)`` [EUR/MWh].
    storage_gwh : numpy.ndarray
        Storage grid ``(K,)`` [GWh]; its last point is 100 %.
    weekly : pandas.DataFrame or None
        Weekly dataset with ``iso_year``, ``iso_week`` and ``storage_gwh`` [GWh]: each year's
        storage at the start of the week is drawn as a thin line.
    ax : matplotlib Axes or None
        Axes to draw on; a new figure is created when None.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _axes(ax, size=(9, 5))
    capacity = storage_gwh[-1]
    weeks = np.arange(1, water_value_eur_mwh.shape[0] + 1)
    mesh = ax.pcolormesh(
        weeks,
        100 * storage_gwh / capacity,
        water_value_eur_mwh.T,
        shading="nearest",
        cmap="viridis",
        vmax=np.percentile(water_value_eur_mwh, HEATMAP_CAP_PERCENTILE),
        rasterized=True,
    )
    bar = ax.figure.colorbar(mesh, ax=ax, pad=0.02, extend="max")
    bar.set_label(
        f"Water value [EUR/MWh] (scale capped at percentile {HEATMAP_CAP_PERCENTILE})",
        color=INK_SECONDARY,
    )
    bar.outline.set_visible(False)
    if weekly is not None:
        drawn = False
        for _, group in weekly.groupby("iso_year"):
            ax.plot(
                group["iso_week"],
                100 * group["storage_gwh"] / capacity,
                color="white",
                linewidth=CONTEXT_WIDTH,
                alpha=0.75,
                label=None if drawn else "observed storage, one line per year",
            )
            drawn = True
        ax.legend(loc="lower left", labelcolor="white")
    ax.grid(False)
    ax.set_xlim(weeks[0] - 0.5, weeks[-1] + 0.5)
    ax.set_ylim(0, 100)
    ax.set_xlabel("ISO week (model stage)")
    ax.set_ylabel(f"Storage at start of week [% of {capacity / 1e3:.1f} TWh]")
    return ax


def water_value_curves(
    water_value_eur_mwh: np.ndarray,
    storage_gwh: np.ndarray,
    weeks: Sequence[int],
    ax: Axes | None = None,
) -> Axes:
    """Water value against storage for selected ISO weeks.

    Parameters
    ----------
    water_value_eur_mwh : numpy.ndarray
        Water values ``(52, K)`` [EUR/MWh].
    storage_gwh : numpy.ndarray
        Storage grid ``(K,)`` [GWh].
    weeks : sequence of int
        ISO weeks to draw (at most eight).
    ax : matplotlib Axes or None
        Axes to draw on; a new figure is created when None.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if len(weeks) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} weeks")
    ax = _axes(ax, size=(7, 4.5))
    filling = 100 * storage_gwh / storage_gwh[-1]
    for week, colour in zip(weeks, SERIES, strict=False):
        ax.plot(filling, water_value_eur_mwh[week - 1], color=colour, label=f"week {week}")
    ax.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Storage at start of week [% of capacity]")
    ax.set_ylabel("Water value [EUR/MWh]")
    ax.legend(loc="upper right")
    return ax


def convergence(
    history_eur_mwh: Sequence[float], tol_eur_mwh: float, ax: Axes | None = None
) -> Axes:
    """Largest change of any water value between consecutive yearly cycles (log scale).

    Parameters
    ----------
    history_eur_mwh : sequence of float
        Change after cycles 2, 3, ... [EUR/MWh].
    tol_eur_mwh : float
        Stopping tolerance [EUR/MWh], drawn as a dashed line.
    ax : matplotlib Axes or None
        Axes to draw on; a new figure is created when None.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _axes(ax, size=(7, 4))
    cycles = np.arange(2, len(history_eur_mwh) + 2)
    ax.semilogy(cycles, history_eur_mwh, marker="o", color=SERIES[0], label="largest change")
    ax.axhline(tol_eur_mwh, color=INK_SECONDARY, linestyle="--", linewidth=1, label="tolerance")
    ax.set_xticks(cycles)
    ax.set_xlabel("Cycle (one pass over the 52 weeks)")
    ax.set_ylabel("Largest water value change [EUR/MWh]")
    ax.legend(loc="upper right")
    return ax


def grid_comparison(
    curves: Mapping[str, tuple[np.ndarray, np.ndarray]], ax: Axes | None = None
) -> Axes:
    """Water value curves of the same week computed on different storage grids.

    Parameters
    ----------
    curves : mapping of str to (numpy.ndarray, numpy.ndarray)
        Label to ``(filling [%], water value [EUR/MWh])`` (at most eight).
    ax : matplotlib Axes or None
        Axes to draw on; a new figure is created when None.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if len(curves) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} curves")
    ax = _axes(ax, size=(7, 4.5))
    styles = ("-", "--", ":", "-.")
    for i, ((label, (filling, value)), colour) in enumerate(
        zip(curves.items(), SERIES, strict=False)
    ):
        ax.plot(filling, value, color=colour, linestyle=styles[i % len(styles)], label=label)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Storage at start of week [% of capacity]")
    ax.set_ylabel("Water value [EUR/MWh]")
    ax.legend(loc="upper right")
    return ax


def backtest_storage(backtest: pd.DataFrame, capacity_gwh: float) -> Figure:
    """Plot simulated and observed end-of-week storage, one panel per backtest year.

    Parameters
    ----------
    backtest : pandas.DataFrame
        Output of :func:`watervalues.simulate.backtest_year` for several years, with
        ``iso_year``, ``iso_week``, ``storage_end_gwh`` and ``observed_storage_end_gwh`` [GWh].
    capacity_gwh : float
        Storage capacity used as 100 % [GWh].

    Returns
    -------
    matplotlib.figure.Figure
    """
    years = sorted(backtest["iso_year"].unique())
    columns = BACKTEST_COLUMNS
    rows = -(-len(years) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(11, 2.3 * rows), sharex=True, sharey=True)
    flat = np.atleast_1d(axes).ravel()
    for ax, year in zip(flat, years, strict=False):
        group = backtest[backtest["iso_year"] == year]
        ax.plot(
            group["iso_week"],
            100 * group["observed_storage_end_gwh"] / capacity_gwh,
            color=INK,
            linewidth=1.2,
            label="observed",
        )
        ax.plot(
            group["iso_week"],
            100 * group["storage_end_gwh"] / capacity_gwh,
            color=SERIES[0],
            linewidth=1.2,
            label="simulated",
        )
        ax.set_title(str(year), fontsize=10)
        ax.set_xlim(1, 52)
        ax.set_ylim(0, 100)
    for ax in flat[len(years) :]:
        ax.set_visible(False)
    flat[0].legend(loc="lower left", fontsize=8)
    fig.supxlabel("ISO week", color=INK_SECONDARY)
    fig.supylabel("Storage at end of week [% of capacity]", color=INK_SECONDARY)
    return fig


def backtest_prices(backtest: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Plot the simulated and observed weekly Swedish price of the backtest [EUR/MWh].

    Parameters
    ----------
    backtest : pandas.DataFrame
        With ``week_start``, ``price_eur_mwh`` and ``observed_price_eur_mwh`` [EUR/MWh].
    ax : matplotlib Axes or None
        Axes to draw on; a new figure is created when None.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _axes(ax)
    x = pd.DatetimeIndex(backtest["week_start"])
    ax.plot(x, backtest["observed_price_eur_mwh"], color=SERIES[0], label="observed")
    ax.plot(x, backtest["price_eur_mwh"], color=SERIES[1], linewidth=1.1, label="simulated")
    ax.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
    ax.set_ylabel("Swedish price [EUR/MWh]")
    ax.legend(loc="upper left", ncols=2)
    return ax


def de_price_scenario(de_price: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Water value along the typical storage path against the annual German price level.

    Parameters
    ----------
    de_price : pandas.DataFrame
        Output of :func:`watervalues.simulate.de_price_scenarios`, indexed by the level
        [EUR/MWh], with ``water_value_mean_eur_mwh``, ``water_value_min_eur_mwh`` and
        ``water_value_max_eur_mwh`` [EUR/MWh].
    ax : matplotlib Axes or None
        Axes to draw on; a new figure is created when None.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _axes(ax, size=(7, 4.5))
    levels = de_price.index.to_numpy(dtype=float)
    ax.fill_between(
        levels,
        de_price["water_value_min_eur_mwh"],
        de_price["water_value_max_eur_mwh"],
        color=CONTEXT,
        alpha=0.6,
        linewidth=0,
        label="range over the weeks",
    )
    ax.plot(
        levels,
        de_price["water_value_mean_eur_mwh"],
        marker="o",
        color=SERIES[0],
        label="mean over the year",
    )
    ax.set_xlabel("Annual mean German price of the scenario [EUR/MWh]")
    ax.set_ylabel("Water value at the typical storage [EUR/MWh]")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    return ax


def dry_and_wet(traces: pd.DataFrame, capacity_gwh: float) -> Figure:
    """Storage and price over the driest and the wettest inflow year from the same start.

    Parameters
    ----------
    traces : pandas.DataFrame
        Output of :func:`watervalues.simulate.dry_and_wet_years`: ``case`` (``dry``/``wet``),
        ``iso_year``, ``iso_week``, ``storage_end_gwh`` [GWh] and ``price_eur_mwh`` [EUR/MWh].
    capacity_gwh : float
        Storage capacity used as 100 % [GWh].

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for case, colour, name in (("wet", SERIES[0], "wettest"), ("dry", SERIES[1], "driest")):
        group = traces[traces["case"] == case]
        label = f"{name} year ({int(group['iso_year'].iloc[0])})"
        top.plot(
            group["iso_week"],
            100 * group["storage_end_gwh"] / capacity_gwh,
            color=colour,
            label=label,
        )
        bottom.plot(group["iso_week"], group["price_eur_mwh"], color=colour, label=label)
    top.set_ylim(0, 100)
    top.set_ylabel("Storage at end of week [% of capacity]")
    top.legend(loc="lower right")
    bottom.set_xlim(1, 52)
    bottom.set_xlabel("ISO week")
    bottom.set_ylabel("Simulated price [EUR/MWh]")
    return fig
