"""Weekly energy inflow of the aggregated reservoir and the inflow scenarios of the SDP.

The inflow follows from the energy balance of the reservoir (guide section 5.1),

    I_w = (S_{w+1} - S_w) + P_w + Sp_w,

with ``S_w`` the storage at the start of week ``w``, ``P_w`` the hydro production and ``Sp_w``
the spill [GWh]. Spill is not observed and is taken as zero, which underestimates the inflow of
very wet weeks. ``P`` covers all hydro, so ``I`` is the total energy inflow (regulated and
unregulated).

Processing order:

1. the balance on unscaled rows: the storage change and the production of a folded week 52 both
   span its 336 hours;
2. negatives within the reporting-noise band (``inflow.negative_noise_gwh``) are clipped to zero
   and listed; a larger negative stops with :class:`InflowError`;
3. rescaling to standard 168-hour stages with :func:`~watervalues.data.aggregate.per_stage`;
4. stages without hydro production (documented gaps) get the mean stage inflow of the same ISO
   week in the other years;
5. the scenario matrix: one row per ISO year (scenario ``j``, probability ``1/J``) and one column
   per stage.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from watervalues.config import PROJECT_ROOT, AppConfig
from watervalues.data.aggregate import per_stage
from watervalues.data.build import GWH_PER_TWH, derived_columns, week_label, write_weekly_csv
from watervalues.units import HOURS_PER_WEEK, MODEL_WEEKS_PER_YEAR

OBSERVED, CLIPPED, IMPUTED = "observed", "clipped", "imputed"
_KEPT_COLUMNS = (
    "week_start",
    "iso_year",
    "iso_week",
    "hours",
    "storage_gwh",
    "storage_end_gwh",
    "hydro_gwh",
)


class InflowError(RuntimeError):
    """A check on the reconstructed inflow failed; the inflow must not be used."""


@dataclass(frozen=True)
class InflowResult:
    """Reconstructed weekly inflow, the scenario matrix and their summaries.

    ``weekly`` is the output of :func:`fill_gaps`, ``scenarios`` of :func:`scenario_matrix`,
    ``climatology`` of :func:`climatology` and ``annual`` of :func:`annual_totals`.
    """

    weekly: pd.DataFrame
    scenarios: pd.DataFrame
    climatology: pd.DataFrame
    annual: pd.DataFrame
    report: list[str]


def _labels(frame: pd.DataFrame) -> list[str]:
    return [week_label(y, k) for y, k in zip(frame["iso_year"], frame["iso_week"], strict=True)]


def reconstruct(weekly: pd.DataFrame, negative_noise_gwh: float) -> pd.DataFrame:
    """Weekly inflow from the reservoir energy balance, per standard 168-hour stage.

    Parameters
    ----------
    weekly : pandas.DataFrame
        Weekly dataset (``weekly_se.csv`` contract): ``hours`` [h], ``storage_gwh`` and
        ``storage_end_gwh`` [GWh], ``hydro_gwh`` [GWh over the row's hours].
    negative_noise_gwh : float
        Width of the storage-reporting noise band [GWh]: a balance in
        ``[-negative_noise_gwh, 0)`` is clipped to zero.

    Returns
    -------
    pandas.DataFrame
        Calendar, storage and hydro columns of ``weekly`` (unchanged) plus ``balance_gwh``, the
        balance over the row's hours [GWh] (NaN where hydro production is missing),
        ``inflow_gwh``, the clipped inflow per 168-hour stage [GWh], and ``status``
        (``observed`` or ``clipped``).

    Raises
    ------
    InflowError
        If a balance is below ``-negative_noise_gwh``: a negative that large is not reporting
        noise, so the reservoir alignment or the production data must be checked.
    """
    balance = derived_columns(weekly)["inflow_gwh"]
    too_negative = balance < -negative_noise_gwh
    if too_negative.any():
        weeks = [
            f"{label} ({value:.1f} GWh)"
            for label, value in zip(
                _labels(weekly[too_negative]), balance[too_negative], strict=True
            )
        ]
        raise InflowError(
            f"weekly inflow below -{negative_noise_gwh:g} GWh is not reporting noise: {weeks}"
        )
    clipped = balance < 0
    out = weekly[list(_KEPT_COLUMNS)].copy()
    out["balance_gwh"] = balance
    out["inflow_gwh"] = balance.where(~clipped, 0.0)
    out = per_stage(out, ["inflow_gwh"])
    out["status"] = np.where(clipped, CLIPPED, OBSERVED)
    return out


def fill_gaps(inflow_weekly: pd.DataFrame) -> pd.DataFrame:
    """Give stages without inflow the mean stage inflow of the same ISO week in the other years.

    Parameters
    ----------
    inflow_weekly : pandas.DataFrame
        Output of :func:`reconstruct`.

    Returns
    -------
    pandas.DataFrame
        A copy in which every NaN ``inflow_gwh`` [GWh per 168-hour stage] is replaced and its
        ``status`` set to ``imputed``. ``balance_gwh`` stays NaN: nothing was measured.

    Raises
    ------
    InflowError
        If an ISO week has no inflow in any year.
    """
    out = inflow_weekly.copy()
    missing = out["inflow_gwh"].isna()
    week_mean = out.groupby("iso_week")["inflow_gwh"].transform("mean")  # skips the gaps
    unfillable = missing & week_mean.isna()
    if unfillable.any():
        raise InflowError(f"no year has an inflow for these weeks: {_labels(out[unfillable])}")
    out.loc[missing, "inflow_gwh"] = week_mean[missing]
    out.loc[missing, "status"] = IMPUTED
    return out


def scenario_matrix(inflow_weekly: pd.DataFrame) -> pd.DataFrame:
    """Inflow scenarios of the SDP: one row per ISO year, one column per weekly stage.

    Parameters
    ----------
    inflow_weekly : pandas.DataFrame
        Output of :func:`fill_gaps`.

    Returns
    -------
    pandas.DataFrame
        Inflow [GWh per 168-hour stage], indexed by ``iso_year`` (ascending; each year is one
        equally likely scenario) with columns ``iso_week`` 1 to 52.

    Raises
    ------
    InflowError
        If a year does not have exactly 52 stages or a stage has no inflow.
    """
    counts = inflow_weekly.groupby("iso_year").size()
    incomplete = counts[counts != MODEL_WEEKS_PER_YEAR]
    if not incomplete.empty:
        raise InflowError(
            f"ISO years without exactly {MODEL_WEEKS_PER_YEAR} stages: {incomplete.to_dict()}"
        )
    missing = inflow_weekly["inflow_gwh"].isna()
    if missing.any():
        raise InflowError(f"stages without inflow: {_labels(inflow_weekly[missing])}")
    matrix = inflow_weekly.pivot(index="iso_year", columns="iso_week", values="inflow_gwh")
    if matrix.columns.tolist() != list(range(1, MODEL_WEEKS_PER_YEAR + 1)) or matrix.isna().any(
        axis=None
    ):
        raise InflowError(f"stages are not ISO weeks 1 to {MODEL_WEEKS_PER_YEAR} in every year")
    return matrix.sort_index()


def climatology(scenarios: pd.DataFrame) -> pd.DataFrame:
    """Minimum, mean and maximum inflow of each stage over the years.

    Parameters
    ----------
    scenarios : pandas.DataFrame
        Output of :func:`scenario_matrix` [GWh per 168-hour stage].

    Returns
    -------
    pandas.DataFrame
        Indexed by ``iso_week`` with columns ``min``, ``mean`` and ``max`` [GWh per stage].
    """
    return scenarios.agg(["min", "mean", "max"]).T


def annual_totals(inflow_weekly: pd.DataFrame) -> pd.DataFrame:
    """Annual inflow and hydro production per ISO year [TWh].

    ``inflow_iso_twh`` and ``hydro_iso_twh`` sum the actual hours of the ISO year (a 53-week
    year counts all 53 weeks), which is comparable with published statistics.
    ``inflow_model_twh`` and ``hydro_model_twh`` sum the 52 standard 168-hour stages that the SDP
    uses, so years compare on equal length. Weeks without hydro production are skipped in the
    hydro totals; ``imputed_weeks`` counts them.

    Parameters
    ----------
    inflow_weekly : pandas.DataFrame
        Output of :func:`fill_gaps`.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``iso_year`` with the four totals [TWh], ``clipped_weeks``, ``imputed_weeks``
        and ``wet_rank`` (1 = largest ``inflow_model_twh``).
    """
    w = inflow_weekly
    to_actual_hours = w["hours"] / HOURS_PER_WEEK
    by_year = w["iso_year"]
    annual = pd.DataFrame(
        {
            "inflow_iso_twh": (w["inflow_gwh"] * to_actual_hours).groupby(by_year).sum(),
            "inflow_model_twh": w["inflow_gwh"].groupby(by_year).sum(),
            "hydro_iso_twh": w["hydro_gwh"].groupby(by_year).sum(),
            "hydro_model_twh": (w["hydro_gwh"] / to_actual_hours).groupby(by_year).sum(),
        }
    ).div(GWH_PER_TWH)
    annual["clipped_weeks"] = (w["status"] == CLIPPED).groupby(by_year).sum()
    annual["imputed_weeks"] = (w["status"] == IMPUTED).groupby(by_year).sum()
    annual["wet_rank"] = annual["inflow_model_twh"].rank(ascending=False, method="min").astype(int)
    annual.index.name = "iso_year"
    return annual


def dry_and_wet(annual: pd.DataFrame) -> tuple[int, int]:
    """Driest and wettest ISO year by inflow over the 52 standard stages.

    Parameters
    ----------
    annual : pandas.DataFrame
        Output of :func:`annual_totals`.

    Returns
    -------
    tuple of int
        ``(driest, wettest)`` ISO years.
    """
    model_year = annual["inflow_model_twh"]
    return int(model_year.idxmin()), int(model_year.idxmax())


def _week_ranges(weeks: Sequence[int]) -> str:
    """Compress sorted week numbers into ranges, e.g. ``17-31, 41``."""
    ranges: list[list[int]] = []
    for week in weeks:
        if ranges and week == ranges[-1][-1] + 1:
            ranges[-1].append(week)
        else:
            ranges.append([week])
    return ", ".join(f"{r[0]}-{r[-1]}" if len(r) > 1 else f"{r[0]}" for r in ranges)


def inflow_report(
    cfg: AppConfig,
    inflow_weekly: pd.DataFrame,
    scenarios: pd.DataFrame,
    clim: pd.DataFrame,
    annual: pd.DataFrame,
) -> list[str]:
    """Human-readable summary of the reconstruction: treated weeks, annual totals, climatology.

    Parameters
    ----------
    cfg : AppConfig
        Configuration (noise band, plausibility band of annual hydro production).
    inflow_weekly, scenarios, clim, annual : pandas.DataFrame
        Outputs of :func:`fill_gaps`, :func:`scenario_matrix`, :func:`climatology` and
        :func:`annual_totals`.

    Returns
    -------
    list of str
        Report lines; energies in GWh per stage and TWh per year.
    """
    w = inflow_weekly
    years = scenarios.index
    lines = [
        "Inflow I = storage_end - storage + hydro [GWh] (energy balance; spill not observed, "
        "taken as 0)",
        f"Scenario matrix: J = {len(years)} ISO years ({years.min()}-{years.max()}) x "
        f"{scenarios.shape[1]} stages [GWh per 168-h stage], probability 1/J each",
    ]

    clipped = w[w["status"] == CLIPPED]
    noise = cfg.inflow.negative_noise_gwh
    lines.append(
        f"\nNegative weekly balances clipped to 0 (noise band -{noise:g} GWh): {len(clipped)}"
    )
    for label, value in zip(_labels(clipped), clipped["balance_gwh"], strict=True):
        lines.append(f"  {label}: {value:.1f} GWh")
    imputed = w[w["status"] == IMPUTED]
    lines.append(
        "Weeks without hydro production, given the mean stage inflow of the same ISO week in "
        f"the other years: {len(imputed)}"
    )
    for label, value in zip(_labels(imputed), imputed["inflow_gwh"], strict=True):
        lines.append(f"  {label}: {value:.1f} GWh")

    lines.append(
        "\nAnnual totals per ISO year [TWh]: actual hours (iso) and 52 standard stages (model)"
    )
    lines.append(annual.round(1).to_string())
    driest, wettest = dry_and_wet(annual)
    model_year, hydro = annual["inflow_model_twh"], annual["hydro_model_twh"]
    lines.append(
        f"  driest {driest} ({model_year[driest]:.1f} TWh), wettest {wettest} "
        f"({model_year[wettest]:.1f} TWh)"
    )
    for name, total in (("inflow", model_year), ("hydro production", hydro)):
        lines.append(
            f"  {name} per model year: mean {total.mean():.1f} TWh, std {total.std():.1f} TWh "
            f"(coefficient of variation {total.std() / total.mean():.1%})"
        )
    band = cfg.data.sanity.hydro_twh_per_year
    outside = model_year[(model_year < band[0]) | (model_year > band[1])]
    status = "OK" if outside.empty else f"outside in {list(outside.index)}"
    lines.append(f"  model-year inflow within the hydro production band {band} TWh: {status}")

    lines.append("\nClimatology of the stage inflow [GWh per stage]")
    mean = clim["mean"]
    for name, week in (("highest", mean.idxmax()), ("lowest", mean.idxmin())):
        row = clim.loc[week]
        lines.append(
            f"  {name} mean: week {week} ({row['mean']:.0f} GWh; over the years "
            f"{row['min']:.0f} to {row['max']:.0f})"
        )
    above = mean[mean > mean.mean()]
    lines.append(
        f"  weeks with a mean above the average stage ({mean.mean():.0f} GWh): "
        f"{_week_ranges(above.index.tolist())}, {above.sum() / mean.sum():.0%} of the mean "
        "annual inflow"
    )
    return lines


def build_inflow(cfg: AppConfig, weekly: pd.DataFrame) -> InflowResult:
    """Reconstruct the weekly inflow, fill the gaps and build the scenario matrix.

    Parameters
    ----------
    cfg : AppConfig
        Configuration with the ``inflow`` section.
    weekly : pandas.DataFrame
        Weekly dataset (``weekly_se.csv`` contract) of whole ISO years.

    Returns
    -------
    InflowResult
        Weekly inflow, scenario matrix, climatology, annual totals and the report.

    Raises
    ------
    InflowError
        If a negative inflow is beyond the noise band or a year is incomplete.
    """
    inflow_weekly = fill_gaps(reconstruct(weekly, cfg.inflow.negative_noise_gwh))
    scenarios = scenario_matrix(inflow_weekly)
    clim = climatology(scenarios)
    annual = annual_totals(inflow_weekly)
    report = inflow_report(cfg, inflow_weekly, scenarios, clim, annual)
    return InflowResult(inflow_weekly, scenarios, clim, annual, report)


def write_outputs(cfg: AppConfig, result: InflowResult) -> None:
    """Write the weekly inflow, the scenario matrix and the report (paths in ``cfg.inflow``)."""
    paths = cfg.inflow
    float_format = cfg.data.csv_float_format
    write_weekly_csv(result.weekly, PROJECT_ROOT / paths.weekly_csv, float_format)
    scenarios_path = PROJECT_ROOT / paths.scenarios_csv
    scenarios_path.parent.mkdir(parents=True, exist_ok=True)
    result.scenarios.to_csv(
        scenarios_path, float_format=float_format, lineterminator="\n", encoding="utf-8"
    )
    report_path = PROJECT_ROOT / paths.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(result.report) + "\n", encoding="utf-8", newline="\n")
