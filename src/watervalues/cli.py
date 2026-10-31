"""Command-line interface: ``watervalues <command>``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from watervalues import __version__
from watervalues.config import load_config
from watervalues.data import filelib

app = typer.Typer(no_args_is_help=True, add_completion=False)

ConfigOption = Annotated[
    Path | None,
    typer.Option(
        "--config",
        "-c",
        help="YAML config file (default: config/base.yaml).",
        exists=True,
        dir_okay=False,
    ),
]


@app.callback()
def main() -> None:
    """Water values for the aggregated Swedish hydro reservoir."""


@app.command()
def info(config: ConfigOption = None) -> None:
    """Show the package version and the validated configuration."""
    cfg = load_config(config)
    typer.echo(f"watervalues {__version__}")
    typer.echo(cfg.model_dump_json(indent=2))


SourceOption = Annotated[
    str | None,
    typer.Option("--source", "-s", help="entsoe_api or filelib (default: data.source in config)."),
]


@app.command()
def fetch(
    dataset: Annotated[
        list[str] | None,
        typer.Option("--dataset", "-d", help="generation, load, price or reservoir (repeatable)."),
    ] = None,
    source: SourceOption = None,
    force: Annotated[bool, typer.Option(help="Request or download again what is cached.")] = False,
    config: ConfigOption = None,
) -> None:
    """Download the raw data and rebuild the normalised cache in data/raw/."""
    from watervalues.config import DATASETS

    cfg = load_config(config)
    chosen = source or cfg.data.source
    datasets = dataset or list(DATASETS)
    if chosen == "entsoe_api":
        from watervalues.data.entsoe_api import ApiError, fetch_api

        try:
            fetch_api(cfg, datasets, force=force, progress=typer.echo)
        except ApiError as exc:
            typer.echo(f"API request failed: {exc}", err=True)
            raise typer.Exit(code=1) from None
    elif chosen == "filelib":
        from watervalues.data.fetch import fetch_filelib

        fetch_filelib(cfg, datasets, force=force, progress=typer.echo)
    else:
        typer.echo(f"Unknown source {chosen!r}: use entsoe_api or filelib.", err=True)
        raise typer.Exit(code=1)


@app.command(name="check-api")
def check_api_command(config: ConfigOption = None) -> None:
    """Verify the ENTSO-E API token with one small request (the token is never shown)."""
    from watervalues.data.entsoe_api import ApiError, check_api
    from watervalues.data.filelib import CredentialsError

    try:
        received = check_api(load_config(config))
    except (ApiError, CredentialsError) as exc:
        typer.echo(f"API check failed: {exc}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(f"API token OK: {received} day-ahead prices received.")


@app.command()
def build(
    strict: Annotated[
        bool, typer.Option(help="Fail on missing values not listed in data/known_gaps.yaml.")
    ] = False,
    source: SourceOption = None,
    config: ConfigOption = None,
) -> None:
    """Build data/processed/weekly_se.csv from the raw cache and print the sanity report."""
    from watervalues.config import PROJECT_ROOT
    from watervalues.data.build import (
        BuildError,
        build_weekly,
        load_known_gaps,
        unexplained_missing,
        write_weekly_csv,
    )

    cfg = load_config(config)
    try:
        result = build_weekly(cfg, source or cfg.data.source)
    except BuildError as exc:
        typer.echo(f"Build stopped: {exc}", err=True)
        raise typer.Exit(code=1) from None
    for line in result.report:
        typer.echo(line)
    missing = unexplained_missing(
        result.weekly, load_known_gaps(PROJECT_ROOT / cfg.data.known_gaps_file)
    )
    if missing:
        typer.echo(f"\n{len(missing)} missing values are not documented: {missing}")
        if strict:
            raise typer.Exit(code=1)
    target = PROJECT_ROOT / cfg.data.weekly_csv
    write_weekly_csv(result.weekly, target, cfg.data.csv_float_format)
    report = PROJECT_ROOT / cfg.data.build_report
    report.write_text("\n".join(result.report) + "\n", encoding="utf-8", newline="\n")
    typer.echo(f"\nWrote {cfg.data.weekly_csv} ({len(result.weekly)} weeks) and the report")


@app.command()
def validate(
    source: SourceOption = None,
    force: Annotated[bool, typer.Option(help="Download the workbooks again.")] = False,
    config: ConfigOption = None,
) -> None:
    """Compare the reservoir data with Energiföretagen's weekly statistics (2019 onwards)."""
    from watervalues.config import PROJECT_ROOT
    from watervalues.data.energiforetagen import WorkbookError, validate_energiforetagen

    cfg = load_config(config)
    try:
        _, lines = validate_energiforetagen(cfg, source or cfg.data.source, force=force)
    except WorkbookError as exc:
        typer.echo(f"Validation failed: {exc}", err=True)
        raise typer.Exit(code=1) from None
    for line in lines:
        typer.echo(line)
    report = PROJECT_ROOT / cfg.data.energiforetagen.report
    report.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    typer.echo(f"\nWrote {cfg.data.energiforetagen.report}")


@app.command()
def inflow(config: ConfigOption = None) -> None:
    """Reconstruct weekly inflows and the scenario matrix from weekly_se.csv (into outputs/)."""
    from watervalues.config import PROJECT_ROOT
    from watervalues.data.build import load_weekly
    from watervalues.inflow import InflowError, build_inflow, write_outputs

    cfg = load_config(config)
    try:
        result = build_inflow(cfg, load_weekly(PROJECT_ROOT / cfg.data.weekly_csv))
    except InflowError as exc:
        typer.echo(f"Inflow stopped: {exc}", err=True)
        raise typer.Exit(code=1) from None
    for line in result.report:
        typer.echo(line)
    write_outputs(cfg, result)
    paths = cfg.inflow
    typer.echo(f"\nWrote {paths.weekly_csv}, {paths.scenarios_csv} and {paths.report}")


@app.command()
def price(config: ConfigOption = None) -> None:
    """Calibrate the price curve and compare its variants from weekly_se.csv (into outputs/)."""
    from watervalues.config import PROJECT_ROOT
    from watervalues.data.build import load_weekly
    from watervalues.price import PriceError, build_price, write_outputs

    cfg = load_config(config)
    try:
        result = build_price(cfg, load_weekly(PROJECT_ROOT / cfg.data.weekly_csv))
    except PriceError as exc:
        typer.echo(f"Price curve stopped: {exc}", err=True)
        raise typer.Exit(code=1) from None
    for line in result.report:
        typer.echo(line)
    write_outputs(cfg, result)
    paths = cfg.price_model
    typer.echo(
        f"\nWrote {paths.weekly_csv}, {paths.variants_csv}, {paths.curve_json} and {paths.report}"
    )


@app.command()
def solve(config: ConfigOption = None) -> None:
    """Solve the water value SDP from weekly_se.csv (into outputs/)."""
    from watervalues.config import PROJECT_ROOT
    from watervalues.data.build import load_weekly
    from watervalues.inflow import InflowError
    from watervalues.price import PriceError
    from watervalues.sdp import SdpError, build_problem, sdp_report, timed_solve, write_outputs

    cfg = load_config(config)
    weekly = load_weekly(PROJECT_ROOT / cfg.data.weekly_csv)
    try:
        solution, seconds = timed_solve(cfg, build_problem(cfg, weekly))
    except (InflowError, PriceError, SdpError) as exc:
        typer.echo(f"SDP stopped: {exc}", err=True)
        raise typer.Exit(code=1) from None
    report = sdp_report(cfg, weekly, solution, seconds)
    for line in report:
        typer.echo(line)
    write_outputs(cfg, solution, report)
    paths = cfg.sdp
    typer.echo(f"\nWrote {paths.values_npz}, {paths.water_values_csv} and {paths.report}")
    if not solution.converged:
        typer.echo("The water values did not converge: raise sdp.max_cycles or check inputs.")
        raise typer.Exit(code=1)


@app.command()
def simulate(config: ConfigOption = None) -> None:
    """Backtest every year and run the scenarios from weekly_se.csv (into outputs/)."""
    from watervalues.config import PROJECT_ROOT
    from watervalues.data.build import load_weekly
    from watervalues.inflow import InflowError
    from watervalues.price import PriceError
    from watervalues.sdp import SdpError
    from watervalues.simulate import SimulationError, run_simulations, write_outputs

    cfg = load_config(config)
    weekly = load_weekly(PROJECT_ROOT / cfg.data.weekly_csv)
    try:
        result = run_simulations(cfg, weekly)
    except (InflowError, PriceError, SdpError, SimulationError) as exc:
        typer.echo(f"Simulation stopped: {exc}", err=True)
        raise typer.Exit(code=1) from None
    for line in result.report:
        typer.echo(line)
    write_outputs(cfg, result)
    typer.echo(f"\nWrote the backtest, its metrics, the scenarios and {cfg.simulation.report}")


@app.command()
def figures(config: ConfigOption = None) -> None:
    """Draw the README figures from outputs/ into reports/figures/."""
    from watervalues.figures import FiguresError, make_figures
    from watervalues.inflow import InflowError
    from watervalues.price import PriceError
    from watervalues.sdp import SdpError

    cfg = load_config(config)
    try:
        written = make_figures(cfg)
    except (FiguresError, InflowError, PriceError, SdpError) as exc:
        typer.echo(f"Figures stopped: {exc}", err=True)
        raise typer.Exit(code=1) from None
    for path in written:
        typer.echo(f"Wrote {path.name}")


@app.command()
def findings(config: ConfigOption = None) -> None:
    """Write the README's key findings from outputs/ (and reports/key_numbers.json)."""
    from watervalues.findings import FindingsError, render, write_findings
    from watervalues.inflow import InflowError
    from watervalues.price import PriceError

    cfg = load_config(config)
    try:
        numbers = write_findings(cfg)
    except (FindingsError, InflowError, PriceError) as exc:
        typer.echo(f"Findings stopped: {exc}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(render(numbers))
    typer.echo(f"\nWrote {cfg.findings.numbers_json} and the key findings of {cfg.findings.readme}")


@app.command()
def notebooks() -> None:
    """Execute every notebook in notebooks/ in order, in place, so outputs are up to date."""
    import nbformat
    from nbclient import NotebookClient

    from watervalues.config import PROJECT_ROOT

    folder = PROJECT_ROOT / "notebooks"
    paths = sorted(folder.glob("[0-9][0-9]_*.ipynb"))
    if not paths:
        typer.echo("No notebooks found.")
        raise typer.Exit(code=1)
    for path in paths:
        typer.echo(f"Executing {path.name} ...")
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=600,
            kernel_name="python3",
            resources={"metadata": {"path": str(folder)}},
        )
        client.execute()
        nbformat.write(notebook, path)
    typer.echo(f"Executed {len(paths)} notebook(s).")


@app.command()
def login(config: ConfigOption = None) -> None:
    """Check your ENTSO-E website login once and store it in the OS credential store.

    Nothing is saved if the login is rejected. The password is never shown or logged.
    """
    settings = load_config(config).data.filelib
    username = typer.prompt("ENTSO-E Transparency Platform email").strip()
    password = typer.prompt(
        "Password (nothing is shown while you type or paste; press Enter)", hide_input=True
    )
    try:
        filelib.check_login(
            username, password, settings.token_url, settings.client_id, settings.timeout_s
        )
    except filelib.LoginError as exc:
        typer.echo(f"Login rejected ({exc}). Nothing was saved.", err=True)
        raise typer.Exit(code=1) from None
    filelib.store_credentials(settings.keyring_service, username, password)
    typer.echo("Login OK. Saved in the OS credential store; earlier entries were removed.")
