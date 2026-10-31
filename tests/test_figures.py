"""Tests for the README figures and key findings (drawn from outputs of a coarse run, offline)."""

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from watervalues import findings as findings_module
from watervalues.cli import app
from watervalues.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from watervalues.data.build import load_weekly
from watervalues.findings import END, START, load_numbers, readme_block, render
from watervalues.price import load_curve
from watervalues.simulate import run_simulations, write_outputs


def _redirect(node, tmp_path):
    """Send every path under outputs/ to the temporary folder, so no test touches outputs/."""
    if isinstance(node, dict):
        return {key: _redirect(value, tmp_path) for key, value in node.items()}
    if isinstance(node, str) and node.startswith("outputs/"):
        return str(tmp_path / node.removeprefix("outputs/"))
    return node


def _config(tmp_path) -> str:
    raw = _redirect(yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")), tmp_path)
    raw["sdp"]["n_storage_points"] = 21  # coarse: the figures do not depend on the grid
    raw["figures"]["dir"] = str(tmp_path / "figures")
    raw["findings"]["numbers_json"] = str(tmp_path / "key_numbers.json")
    raw["findings"]["readme"] = str(tmp_path / "README.md")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(path)


def test_no_output_path_is_left_in_outputs(tmp_path) -> None:
    _config(tmp_path)
    assert "outputs/" not in (tmp_path / "config.yaml").read_text(encoding="utf-8")


def test_figures_need_the_outputs(tmp_path) -> None:
    result = CliRunner().invoke(app, ["figures", "--config", _config(tmp_path)])
    assert result.exit_code == 1
    assert "watervalues solve" in result.output


def test_figures_and_findings_are_drawn_from_the_outputs(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path)
    runner = CliRunner()
    for command in ("price", "solve"):
        result = runner.invoke(app, [command, "--config", config])
        assert result.exit_code == 0, result.output
    # One backtest year is enough to draw every figure (`watervalues simulate` runs all).
    cfg = load_config(config)
    weekly = load_weekly(PROJECT_ROOT / cfg.data.weekly_csv)
    simulated = run_simulations(cfg, weekly, years=[2019])
    write_outputs(cfg, simulated)
    result = runner.invoke(app, ["figures", "--config", config])
    assert result.exit_code == 0, result.output
    pngs = sorted(path.name for path in (tmp_path / "figures").glob("*.png"))
    assert pngs == [
        "backtest_prices.png",
        "backtest_storage.png",
        "price_slope_by_year.png",
        "scenario_de_price.png",
        "scenario_dry_wet.png",
        "water_values.png",
    ]
    assert all((tmp_path / "figures" / name).stat().st_size > 10_000 for name in pngs)

    (tmp_path / "README.md").write_text(f"# x\n{START}\n{END}\n", encoding="utf-8")
    # A coarse one-year run need not agree with the README's wording (tests/test_findings.py
    # tests that guard); here only the path from outputs to numbers is tested.
    monkeypatch.setattr(findings_module, "_check", lambda numbers: None)
    result = runner.invoke(app, ["findings", "--config", config])
    assert result.exit_code == 0, result.output
    numbers = load_numbers(tmp_path / "key_numbers.json")
    assert readme_block((tmp_path / "README.md").read_text(encoding="utf-8")) == render(numbers)
    # The numbers are what the outputs say.
    curve = load_curve(cfg.price_model.curve_json)
    expected = curve.gamma + curve.kappa_per_gwh * numbers["mean_net_import_gwh"]
    assert numbers["pass_through_at_mean_import"] == pytest.approx(expected, abs=1e-6)
    assert numbers["years"] == "2019-2019"  # the years actually replayed
    assert numbers["de_price_slope_typical"] == pytest.approx(simulated.de_price_slope, rel=1e-3)
    dry = simulated.dry_wet[simulated.dry_wet["case"] == "dry"]
    assert numbers["dry_end_storage_twh"] == pytest.approx(
        dry["storage_end_gwh"].iloc[-1] / 1_000, abs=1e-3
    )
    assert np.isfinite(numbers["de_price_slope_scarce"])
