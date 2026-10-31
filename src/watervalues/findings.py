"""Key findings for the README, read from the model outputs (never typed by hand).

``watervalues findings`` collects a few numbers from ``outputs/`` into
``reports/key_numbers.json``, which is committed, and writes the README's key-findings block
from that file. A test checks that the README block is exactly what the file renders, so the
README states what the last run produced. The wording asserts directions (for example that the
simulated prices are too low); :func:`key_numbers` stops if a run contradicts one of them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from watervalues.config import PROJECT_ROOT, AppConfig
from watervalues.price import USED, CurveSpec, load_curve
from watervalues.sdp import load_solution, water_value_at
from watervalues.simulate import load_metrics
from watervalues.units import GWH_PER_TWH

START = "<!-- key-findings:start -->"
END = "<!-- key-findings:end -->"
DECIMALS = 6  # numbers are stored rounded, so that a rerun on another platform matches


class FindingsError(RuntimeError):
    """An output is missing, contradicts the wording, or the README has no key-findings block."""


def _require(path: Path, command: str) -> Path:
    if not path.exists():
        raise FindingsError(f"{path} is missing: run `watervalues {command}` first")
    return path


def _slope(levels: pd.Index, values: pd.Series) -> float:
    """Least-squares slope of values against levels [per unit of level]."""
    return float(np.polyfit(levels.to_numpy(dtype=float), values.to_numpy(dtype=float), 1)[0])


def _check(numbers: Mapping[str, float | int | str]) -> None:
    """Stop if the outputs contradict a direction that the README's wording asserts."""
    n = numbers
    assertions = {
        "water is worth more when scarce than when full": n["water_value_scarce_eur_mwh"]
        > n["water_value_full_eur_mwh"],
        "the base price curve beats the constant slope out of sample": n["loyo_rmse_base_eur_mwh"]
        < n["loyo_rmse_constant_eur_mwh"],
        "the simulated prices are too low": n["backtest_price_bias_eur_mwh"] < 0,
        "the operating effect lowers the price": n["backtest_operating_bias_eur_mwh"] < 0,
        "the dry year is dearer than the wet one": n["dry_price_mean_eur_mwh"]
        > n["wet_price_mean_eur_mwh"],
        "the dry year ends with less water": n["dry_end_storage_twh"] < n["wet_end_storage_twh"],
    }
    failed = [claim for claim, holds in assertions.items() if not holds]
    if failed:
        raise FindingsError(f"the outputs contradict the README's wording: {failed}")


def key_numbers(cfg: AppConfig) -> dict[str, float | int | str]:
    """Collect the numbers quoted in the README from the outputs of the pipeline.

    Parameters
    ----------
    cfg : AppConfig
        Configuration: output paths of ``price_model``, ``sdp`` and ``simulation``, and the
        ``findings`` section (the storage points quoted).

    Returns
    -------
    dict
        Water values [EUR/MWh] at the quoted points, the price curve's pass-through of the
        German price at balanced trade and at the mean net import [-], out-of-sample price
        errors [EUR/MWh], continental price slopes along the typical path and at the scarce
        point [EUR/MWh per EUR/MWh], backtest metrics, and the dry and wet years with their mean
        prices [EUR/MWh] and final storage [TWh]; floats rounded to ``DECIMALS``.

    Raises
    ------
    FindingsError
        If an output of ``watervalues price``, ``solve`` or ``simulate`` is missing, or the
        outputs contradict a direction the README's wording asserts.
    """
    pm, sdp, sim, found = cfg.price_model, cfg.sdp, cfg.simulation, cfg.findings
    curve = load_curve(_require(PROJECT_ROOT / pm.curve_json, "price"))
    variants = pd.read_csv(_require(PROJECT_ROOT / pm.variants_csv, "price"), index_col="label")
    table = pd.read_csv(_require(PROJECT_ROOT / pm.weekly_csv, "price"))
    solution = load_solution(_require(PROJECT_ROOT / sdp.values_npz, "solve"))
    metrics = load_metrics(_require(PROJECT_ROOT / sim.metrics_csv, "simulate"))
    de_price = pd.read_csv(
        _require(PROJECT_ROOT / sim.de_price_csv, "simulate"), index_col="de_price_level_eur_mwh"
    )
    dry_wet = pd.read_csv(_require(PROJECT_ROOT / sim.dry_wet_csv, "simulate"))

    constant = CurveSpec("constant", curve.spec.seasons, curve.spec.exclude_years).label
    if constant == curve.spec.label:
        raise FindingsError("the base price curve has a constant slope: nothing to compare")
    mean_import = float(table.loc[table["status"] == USED, "net_import_stage_gwh"].mean())
    pooled = metrics.loc["all"]
    years = [int(year) for year in metrics.index if year != "all"]
    cases = dict(list(dry_wet.groupby("case")))
    scarce_week, scarce_pct = found.scarce_point
    full_week, full_pct = found.full_point
    numbers: dict[str, float | int | str] = {
        "years": f"{min(years)}-{max(years)}",
        "scarce_week": int(scarce_week),
        "scarce_filling_pct": float(scarce_pct),
        "water_value_scarce_eur_mwh": water_value_at(solution, scarce_week, scarce_pct),
        "full_week": int(full_week),
        "full_filling_pct": float(full_pct),
        "water_value_full_eur_mwh": water_value_at(solution, full_week, full_pct),
        "gamma": curve.gamma,
        "mean_net_import_gwh": mean_import,
        "pass_through_at_mean_import": float(curve.de_pass_through(mean_import)),
        "loyo_rmse_base_eur_mwh": float(variants.loc[curve.spec.label, "loyo_rmse_eur_mwh"]),
        "loyo_rmse_constant_eur_mwh": float(variants.loc[constant, "loyo_rmse_eur_mwh"]),
        "de_price_slope_typical": _slope(de_price.index, de_price["water_value_mean_eur_mwh"]),
        "de_price_slope_scarce": _slope(de_price.index, de_price["water_value_scarce_eur_mwh"]),
        "backtest_storage_rmse_pct": float(pooled["storage_rmse_pct"]),
        "backtest_price_corr": float(pooled["price_corr"]),
        "backtest_price_corr_annual_means": float(pooled["price_corr_annual_means"]),
        "backtest_price_corr_within_year": float(pooled["price_corr_within_year"]),
        "backtest_price_bias_eur_mwh": float(pooled["price_bias_eur_mwh"]),
        "backtest_operating_bias_eur_mwh": float(pooled["operating_bias_eur_mwh"]),
        "backtest_curve_bias_eur_mwh": float(pooled["curve_bias_eur_mwh"]),
        "dry_year": int(cases["dry"]["iso_year"].iloc[0]),
        "wet_year": int(cases["wet"]["iso_year"].iloc[0]),
        "dry_price_mean_eur_mwh": float(cases["dry"]["price_eur_mwh"].mean()),
        "wet_price_mean_eur_mwh": float(cases["wet"]["price_eur_mwh"].mean()),
        "dry_end_storage_twh": float(cases["dry"]["storage_end_gwh"].iloc[-1] / GWH_PER_TWH),
        "wet_end_storage_twh": float(cases["wet"]["storage_end_gwh"].iloc[-1] / GWH_PER_TWH),
    }
    numbers = {k: round(v, DECIMALS) if isinstance(v, float) else v for k, v in numbers.items()}
    _check(numbers)
    return numbers


def render(numbers: Mapping[str, float | int | str]) -> str:
    """Render the README's key-findings block as Markdown bullets from :func:`key_numbers`."""
    n = numbers
    price_gap = n["dry_price_mean_eur_mwh"] - n["wet_price_mean_eur_mwh"]
    bullets = [
        f"Water is worth most when the reservoir is low before the winter: for example about "
        f"{n['water_value_scarce_eur_mwh']:.0f} EUR/MWh in week {n['scarce_week']} at "
        f"{n['scarce_filling_pct']:g}% storage, against {n['water_value_full_eur_mwh']:.0f} "
        f"EUR/MWh in week {n['full_week']} with a {n['full_filling_pct']:g}% full reservoir just "
        "before the spring flood.",
        f"The Swedish price follows the German price almost one for one at balanced trade "
        f"(pass-through {n['gamma']:.2f}), and the discount that exports buy grows with the "
        f"German price. At the average net import the Swedish price moves "
        f"{n['pass_through_at_mean_import']:.2f} EUR/MWh per EUR/MWh in Germany. Letting the "
        f"price slope scale with the German price cuts the out-of-sample price error from "
        f"{n['loyo_rmse_constant_eur_mwh']:.1f} to {n['loyo_rmse_base_eur_mwh']:.1f} EUR/MWh.",
        f"Along the typical storage path, each EUR/MWh on the continental price moves the "
        f"Swedish water value by {n['de_price_slope_typical']:.2f} EUR/MWh on average over the "
        f"year; with a low reservoir in week {n['scarce_week']} the response is "
        f"{n['de_price_slope_scarce']:.2f} EUR/MWh.",
        f"Replaying {n['years']} with water values that knew each year's demand and German "
        f"price but not its inflow, the storage error is {n['backtest_storage_rmse_pct']:.1f}% "
        f"of the largest storage observed (RMSE), and the weekly price correlates "
        f"{n['backtest_price_corr']:.2f} with the observed one "
        f"({n['backtest_price_corr_annual_means']:.2f} for annual means, "
        f"{n['backtest_price_corr_within_year']:.2f} within years). The simulated prices run "
        f"{-n['backtest_price_bias_eur_mwh']:.1f} EUR/MWh low on average: "
        f"{-n['backtest_operating_bias_eur_mwh']:.1f} from releasing more water than the "
        f"producers did, and {-n['backtest_curve_bias_eur_mwh']:.1f} from the price curve at the "
        "observed net import.",
        f"From the same start, the driest year ({n['dry_year']}) ends with "
        f"{n['dry_end_storage_twh']:.1f} TWh in storage against {n['wet_end_storage_twh']:.1f} "
        f"TWh for the wettest ({n['wet_year']}), at a mean price {price_gap:.1f} EUR/MWh higher.",
    ]
    return "\n".join(f"- {bullet}" for bullet in bullets)


def readme_block(text: str) -> str:
    """Return the content between the key-findings markers of a README.

    Raises
    ------
    FindingsError
        If the markers are missing.
    """
    if START not in text or END not in text:
        raise FindingsError(f"the README has no {START} ... {END} block")
    return text.split(START, 1)[1].split(END, 1)[0].strip()


def update_readme(text: str, block: str) -> str:
    """Replace the content between the key-findings markers."""
    readme_block(text)  # checks the markers
    head, rest = text.split(START, 1)
    tail = rest.split(END, 1)[1]
    return f"{head}{START}\n{block}\n{END}{tail}"


def write_findings(cfg: AppConfig) -> dict[str, float | int | str]:
    """Write ``findings.numbers_json`` and the README block; return the numbers."""
    numbers = key_numbers(cfg)
    json_path = PROJECT_ROOT / cfg.findings.numbers_json
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(numbers, indent=2) + "\n", encoding="utf-8", newline="\n")
    readme = PROJECT_ROOT / cfg.findings.readme
    text = readme.read_text(encoding="utf-8")
    readme.write_text(update_readme(text, render(numbers)), encoding="utf-8", newline="\n")
    return numbers


def load_numbers(path: Path) -> dict[str, float | int | str]:
    """Read the numbers written by :func:`write_findings`."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
