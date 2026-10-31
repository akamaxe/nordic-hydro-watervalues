"""Tests for the README figures (drawn from outputs written by a coarse run, offline)."""

import yaml
from typer.testing import CliRunner

from watervalues.cli import app
from watervalues.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from watervalues.data.build import load_weekly
from watervalues.findings import END, START, load_numbers, readme_block, render
from watervalues.simulate import run_simulations, write_outputs


def _config(tmp_path) -> str:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["sdp"].update(
        n_storage_points=21,  # coarse: the figures do not depend on the grid
        values_npz=str(tmp_path / "water_values.npz"),
        water_values_csv=str(tmp_path / "water_values.csv"),
        report=str(tmp_path / "sdp_report.txt"),
    )
    for name in ("backtest_csv", "metrics_csv", "de_price_csv", "dry_wet_csv"):
        raw["simulation"][name] = str(tmp_path / f"{name}.csv")
    raw["simulation"]["report"] = str(tmp_path / "simulate_report.txt")
    raw["figures"]["dir"] = str(tmp_path / "figures")
    raw["findings"]["numbers_json"] = str(tmp_path / "key_numbers.json")
    raw["findings"]["readme"] = str(tmp_path / "README.md")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(path)


def test_figures_need_the_outputs(tmp_path) -> None:
    result = CliRunner().invoke(app, ["figures", "--config", _config(tmp_path)])
    assert result.exit_code == 1
    assert "watervalues solve" in result.output


def test_figures_are_drawn_from_the_outputs(tmp_path) -> None:
    config = _config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["solve", "--config", config])
    assert result.exit_code == 0, result.output
    # One backtest year is enough to draw every figure (`watervalues simulate` runs all).
    cfg = load_config(config)
    weekly = load_weekly(PROJECT_ROOT / cfg.data.weekly_csv)
    write_outputs(cfg, run_simulations(cfg, weekly, years=[2019]))
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
    # The key findings of a README are written from the same outputs.
    (tmp_path / "README.md").write_text(f"# x\n{START}\n{END}\n", encoding="utf-8")
    result = runner.invoke(app, ["findings", "--config", config])
    assert result.exit_code == 0, result.output
    numbers = load_numbers(tmp_path / "key_numbers.json")
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert readme_block(readme) == render(numbers)
    assert numbers["dry_year"] != numbers["wet_year"]
