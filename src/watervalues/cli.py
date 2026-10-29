"""Command-line interface: ``watervalues <command>``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from watervalues import __version__
from watervalues.config import load_config

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
