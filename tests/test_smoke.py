"""Smoke tests: the package imports, the config validates and the CLI starts."""

import socket

import pytest
from typer.testing import CliRunner

import watervalues
from watervalues.cli import app
from watervalues.config import load_config


def test_version_is_set() -> None:
    assert watervalues.__version__ != "0.0.0"


def test_base_config_validates() -> None:
    cfg = load_config()
    assert cfg.area == "SE"
    assert 0 < cfg.reservoir.availability <= 1
    assert cfg.period.start < cfg.period.end


def test_cli_help_lists_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "info" in result.output


def test_cli_info_prints_config() -> None:
    result = CliRunner().invoke(app, ["info"])
    assert result.exit_code == 0, result.output
    assert "n_storage_points" in result.output


def test_network_is_blocked() -> None:
    with pytest.raises(RuntimeError, match="blocked"):
        socket.create_connection(("example.com", 443), timeout=1)
