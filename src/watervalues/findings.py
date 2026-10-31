"""Key findings for the README, read from the model outputs (never typed by hand).

``watervalues findings`` collects a few numbers from ``outputs/`` (and the price curve, refitted
in a second) into ``reports/key_numbers.json``, which is committed, and writes the README's
key-findings block from that file. A test checks that the README block is exactly what the
file renders, so the README states what the last run produced.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from watervalues.config import PROJECT_ROOT, AppConfig
from watervalues.data.build import load_weekly
from watervalues.price import USED, CurveSpec, build_price
from watervalues.sdp import load_solution, water_value_at
from watervalues.simulate import load_metrics
from watervalues.units import GWH_PER_TWH

START = "<!-- key-findings:start -->"
END = "<!-- key-findings:end -->"


class FindingsError(RuntimeError):
    """An output is missing or the README has no key-findings block."""


def _require(path: Path, command: str) -> Path:
    if not path.exists():
        raise FindingsError(f"{path} is missing: run `watervalues {command}` first")
    return path


def key_numbers(cfg: AppConfig) -> dict[str, float | int | str]:
    """Collect the numbers quoted in the README from the outputs of the pipeline.

    Parameters
    ----------
    cfg : AppConfig
        Configuration: output paths of ``sdp`` and ``simulation``, and the ``findings``
        section (the storage points quoted).

    Returns
    -------
    dict
        Water values [EUR/MWh] at the quoted points, price curve coefficients and the
        pass-through at the mean net import [-], out-of-sample price errors [EUR/MWh], the
        continental price slope [EUR/MWh per EUR/MWh], backtest metrics, and the dry and wet
        years with their mean prices [EUR/MWh] and final storage [TWh].

    Raises
    ------
    FindingsError
        If an output of ``watervalues solve`` or ``watervalues simulate`` is missing.
    """
    sdp, sim, found = cfg.sdp, cfg.simulation, cfg.findings
    solution = load_solution(_require(PROJECT_ROOT / sdp.values_npz, "solve"))
    metrics = load_metrics(_require(PROJECT_ROOT / sim.metrics_csv, "simulate"))
    de_price = pd.read_csv(
        _require(PROJECT_ROOT / sim.de_price_csv, "simulate"), index_col="de_price_level_eur_mwh"
    )
    dry_wet = pd.read_csv(_require(PROJECT_ROOT / sim.dry_wet_csv, "simulate"))
    price = build_price(cfg, load_weekly(PROJECT_ROOT / cfg.data.weekly_csv))

    base = price.base
    constant = CurveSpec("constant", base.spec.seasons, base.spec.exclude_years).label
    used = price.table[price.table["status"] == USED]
    mean_import = float(used["net_import_stage_gwh"].mean())
    pooled = metrics.loc["all"]
    cases = dict(list(dry_wet.groupby("case")))
    scarce_week, scarce_pct = found.scarce_point
    full_week, full_pct = found.full_point
    return {
        "years": f"{cfg.period.first_iso_year}-{cfg.period.last_iso_year}",
        "scarce_week": int(scarce_week),
        "scarce_filling_pct": float(scarce_pct),
        "water_value_scarce_eur_mwh": water_value_at(solution, scarce_week, scarce_pct),
        "full_week": int(full_week),
        "full_filling_pct": float(full_pct),
        "water_value_full_eur_mwh": water_value_at(solution, full_week, full_pct),
        "gamma": base.gamma,
        "mean_net_import_gwh": mean_import,
        "pass_through_at_mean_import": float(base.de_pass_through(mean_import)),
        "loyo_rmse_base_eur_mwh": float(price.comparison.loc[base.spec.label, "loyo_rmse_eur_mwh"]),
        "loyo_rmse_constant_eur_mwh": float(price.comparison.loc[constant, "loyo_rmse_eur_mwh"]),
        "de_price_slope": float(
            np.polyfit(de_price.index, de_price["water_value_mean_eur_mwh"], 1)[0]
        ),
        "backtest_storage_rmse_pct": float(pooled["storage_rmse_pct"]),
        "backtest_price_corr": float(pooled["price_corr"]),
        "backtest_price_corr_annual_means": float(pooled["price_corr_annual_means"]),
        "backtest_price_corr_within_year": float(pooled["price_corr_within_year"]),
        "backtest_price_bias_eur_mwh": float(pooled["price_bias_eur_mwh"]),
        "backtest_operating_bias_eur_mwh": float(pooled["operating_bias_eur_mwh"]),
        "dry_year": int(cases["dry"]["iso_year"].iloc[0]),
        "wet_year": int(cases["wet"]["iso_year"].iloc[0]),
        "dry_price_mean_eur_mwh": float(cases["dry"]["price_eur_mwh"].mean()),
        "wet_price_mean_eur_mwh": float(cases["wet"]["price_eur_mwh"].mean()),
        "dry_end_storage_twh": float(cases["dry"]["storage_end_gwh"].iloc[-1] / GWH_PER_TWH),
        "wet_end_storage_twh": float(cases["wet"]["storage_end_gwh"].iloc[-1] / GWH_PER_TWH),
    }


def render(numbers: Mapping[str, float | int | str]) -> str:
    """Render the README's key-findings block as Markdown bullets from :func:`key_numbers`."""
    n = numbers
    price_gap = n["dry_price_mean_eur_mwh"] - n["wet_price_mean_eur_mwh"]
    bullets = [
        f"Water is worth most when the reservoir is low before the winter: about "
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
        f"Each EUR/MWh on the continental price moves the Swedish water value by "
        f"{n['de_price_slope']:.2f} EUR/MWh.",
        f"Replaying {n['years']} with water values that knew each year's demand and German "
        f"price but not its inflow, the storage error is {n['backtest_storage_rmse_pct']:.1f}% "
        f"of the largest storage observed (RMSE), and the weekly price correlates "
        f"{n['backtest_price_corr']:.2f} with the observed one "
        f"({n['backtest_price_corr_annual_means']:.2f} for annual means, "
        f"{n['backtest_price_corr_within_year']:.2f} within years). The model releases more "
        f"water than producers did, so its prices run {-n['backtest_price_bias_eur_mwh']:.1f} "
        "EUR/MWh low on average.",
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
