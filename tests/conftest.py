"""Shared fixtures: keep the test suite offline and free of credentials."""

import socket

import keyring
import pytest

_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Block outbound connections unless the test is marked ``network``."""
    if request.node.get_closest_marker("network"):
        return
    real_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: object) -> None:
        host = address[0] if isinstance(address, tuple) else address
        if host not in _LOOPBACK_HOSTS:
            raise RuntimeError(f"network access is blocked in tests: {address!r}")
        real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest.fixture(autouse=True)
def _no_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide ENTSO-E credentials: environment variables and the OS credential store."""
    for name in ("ENTSOE_API_KEY", "ENTSOE_USERNAME", "ENTSOE_PWD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(keyring, "get_credential", lambda service, username: None)
    monkeypatch.setattr(keyring, "get_password", lambda service, username: None)
    # Never read the project's real .env (it holds the API token).
    from watervalues.data import entsoe_api

    monkeypatch.setattr(entsoe_api, "load_dotenv", lambda *args, **kwargs: False)
