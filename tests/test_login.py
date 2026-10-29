"""Tests for checking and storing the ENTSO-E website login (fake server and credential store)."""

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from watervalues.cli import app
from watervalues.data import filelib
from watervalues.data.filelib import LoginError, check_login, clear_credentials, store_credentials

SERVICE = "svc"


class FakeResponse:
    def __init__(self, status_code: int, body: dict) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> dict:
        return self._body


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.posts: list[dict] = []

    def post(self, url: str, data: dict, timeout: float) -> FakeResponse:
        self.posts.append({"url": url, **data})
        return self.response


def test_check_login_accepts_a_valid_login() -> None:
    session = FakeSession(FakeResponse(200, {"access_token": "t", "expires_in": 900}))
    check_login("me@example.com", "pw", "https://idp/token", "tp-fms-public", 5, session)
    assert session.posts[0]["grant_type"] == "password"
    assert session.posts[0]["client_id"] == "tp-fms-public"


def test_check_login_reports_the_server_reason() -> None:
    body = {"error": "invalid_grant", "error_description": "Invalid user credentials"}
    session = FakeSession(FakeResponse(401, body))
    with pytest.raises(LoginError, match="Invalid user credentials"):
        check_login("me@example.com", "wrong", "https://idp/token", "c", 5, session)


class FakeVault:
    """Mimics keyring on Windows: saving over an entry renames the old one to user@service."""

    def __init__(self) -> None:
        self.entries: dict[str, tuple[str, str]] = {}

    def get_credential(self, service, username):
        entry = self.entries.get(service)
        return None if entry is None else SimpleNamespace(username=entry[0], password=entry[1])

    def set_password(self, service, username, password):
        if service in self.entries:
            old_user, old_password = self.entries[service]
            self.entries[f"{old_user}@{service}"] = (old_user, old_password)
        self.entries[service] = (username, password)

    def delete_password(self, service, username):
        for target in (service, f"{username}@{service}"):
            if target in self.entries and self.entries[target][0] == username:
                del self.entries[target]

    def renamed(self, service):
        return [t for t in self.entries if t.endswith(f"@{service}")]

    def delete_entry(self, target):
        del self.entries[target]


@pytest.fixture
def vault(monkeypatch) -> FakeVault:
    fake = FakeVault()
    monkeypatch.setattr(filelib.keyring, "get_credential", fake.get_credential)
    monkeypatch.setattr(filelib.keyring, "set_password", fake.set_password)
    monkeypatch.setattr(filelib.keyring, "delete_password", fake.delete_password)
    monkeypatch.setattr(filelib, "_renamed_previous_entries", fake.renamed)
    monkeypatch.setattr(filelib, "_delete_entry", fake.delete_entry)
    return fake


def test_store_replaces_every_earlier_entry(vault: FakeVault) -> None:
    vault.set_password(SERVICE, "typo@example.com", "first")
    vault.set_password(SERVICE, "me@example.com", "second")  # renames the first entry
    assert len(vault.entries) == 2
    store_credentials(SERVICE, "me@example.com", "right")
    assert vault.entries == {SERVICE: ("me@example.com", "right")}


def test_clear_counts_deleted_entries(vault: FakeVault) -> None:
    vault.set_password(SERVICE, "a@example.com", "1")
    vault.set_password(SERVICE, "b@example.com", "2")
    assert clear_credentials(SERVICE) == 2
    assert vault.entries == {}


def test_cli_login_saves_only_a_verified_login(vault: FakeVault, monkeypatch) -> None:
    monkeypatch.setattr(filelib, "check_login", lambda *args, **kwargs: None)
    result = CliRunner().invoke(app, ["login"], input="me@example.com\nright\n")
    assert result.exit_code == 0, result.output
    assert vault.entries[filelib_service()] == ("me@example.com", "right")
    assert "right" not in result.output  # the password is never echoed


def test_cli_login_saves_nothing_when_rejected(vault: FakeVault, monkeypatch) -> None:
    def reject(*args, **kwargs):
        raise LoginError("HTTP 401: Invalid user credentials")

    monkeypatch.setattr(filelib, "check_login", reject)
    vault.set_password(filelib_service(), "me@example.com", "old")
    result = CliRunner().invoke(app, ["login"], input="me@example.com\nwrong\n")
    assert result.exit_code == 1
    assert "Invalid user credentials" in result.output
    assert vault.entries == {filelib_service(): ("me@example.com", "old")}


def filelib_service() -> str:
    from watervalues.config import load_config

    return load_config().data.filelib.keyring_service
