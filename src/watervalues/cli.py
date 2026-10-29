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


@app.command()
def fetch(
    dataset: Annotated[
        list[str] | None,
        typer.Option("--dataset", "-d", help="generation, load, price or reservoir (repeatable)."),
    ] = None,
    force: Annotated[bool, typer.Option(help="Download again even if cached.")] = False,
    config: ConfigOption = None,
) -> None:
    """Download the raw data and rebuild the normalised cache in data/raw/."""
    from watervalues.config import DATASETS
    from watervalues.data.fetch import fetch_filelib

    cfg = load_config(config)
    if cfg.data.source != "filelib":
        typer.echo("Only the File Library source is available until the API token arrives.")
        raise typer.Exit(code=1)
    fetch_filelib(cfg, dataset or list(DATASETS), force=force, progress=typer.echo)


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
