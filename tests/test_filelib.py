"""Tests for the File Library transport: credentials, politeness, file selection and caching."""

import io
import zipfile
from types import SimpleNamespace

import pytest
import requests

from watervalues.data import filelib
from watervalues.data.filelib import (
    CredentialsError,
    Manifest,
    RateLimitedError,
    RateLimiter,
    archive_path,
    call_with_retries,
    download_archives,
    file_period,
    get_credentials,
    select_files,
)


def zip_bytes(name: str = "export.csv", text: str = "a\tb\n1\t2\n") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr(name, text)
    return buffer.getvalue()


def http_error(status: int) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(response=response)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_rate_limiter_spaces_calls() -> None:
    clock = FakeClock()
    limiter = RateLimiter(min_interval_s=2.0, clock=clock, sleep=clock.sleep)
    limiter.wait()
    limiter.wait()
    clock.now += 5.0
    limiter.wait()
    assert clock.sleeps == [2.0]


def no_wait() -> RateLimiter:
    clock = FakeClock()
    return RateLimiter(min_interval_s=0.0, clock=clock, sleep=clock.sleep)


def test_server_errors_are_retried_with_backoff() -> None:
    outcomes = [http_error(503), requests.ConnectionError(), "ok"]
    sleeps: list[float] = []

    def flaky() -> str:
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    result = call_with_retries(flaky, no_wait(), max_retries=3, backoff_s=1.0, sleep=sleeps.append)
    assert result == "ok"
    assert sleeps == [1.0, 2.0]


def test_rate_limit_answer_is_not_retried() -> None:
    calls = []

    def limited() -> None:
        calls.append(1)
        raise http_error(429)

    with pytest.raises(RateLimitedError):
        call_with_retries(limited, no_wait(), max_retries=3, backoff_s=0.0, sleep=lambda _: None)
    assert len(calls) == 1


def test_client_errors_are_raised_immediately() -> None:
    def bad_request() -> None:
        raise http_error(400)

    with pytest.raises(requests.HTTPError):
        call_with_retries(bad_request, no_wait(), max_retries=3, backoff_s=0.0)


def test_file_period_and_selection() -> None:
    assert file_period("2018_08_ActualTotalLoad_6.1.A_r3.csv") == (2018, 8)
    assert file_period("2015_AggregatedFillingRate_16.1.D_r3.csv") == (2015, None)
    monthly = ["2014_11_X.csv", "2014_12_X.csv", "2025_12_X.csv", "2026_01_X.csv", "log_r3.csv"]
    assert select_files(monthly, (2014, 12), (2025, 12)) == ["2014_12_X.csv", "2025_12_X.csv"]
    yearly = ["2013_Y.csv", "2014_Y.csv", "2026_Y.csv", "2027_Y.csv"]
    # Yearly files match on their year alone.
    assert select_files(yearly, (2014, 12), (2026, 1)) == ["2014_Y.csv", "2026_Y.csv"]


class FakeFileClient:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.downloads: list[tuple[str, str]] = []

    def list_folder(self, folder: str) -> dict[str, str]:
        return {"2015_01_X.csv": "id1"}

    def download_single_file_raw(self, folder: str, filename: str) -> bytes:
        self.downloads.append((folder, filename))
        return self.payload


def test_download_archives_caches_and_records_manifest(tmp_path) -> None:
    client = FakeFileClient(zip_bytes())
    args = (client, "Folder", ["2015_01_X.csv"], tmp_path, no_wait(), 0, 0.0)
    first = download_archives(*args)
    second = download_archives(*args)
    assert first == second == [archive_path(tmp_path, "Folder", "2015_01_X.csv")]
    assert len(client.downloads) == 1  # the second call used the cache
    manifest = Manifest.load(tmp_path / "filelib" / "_manifest.json")
    assert "Folder/2015_01_X.csv" in manifest.entries
    assert len(manifest.digest()) == 64
    download_archives(*args, force=True)
    assert len(client.downloads) == 2


def test_download_rejects_non_zip_answers(tmp_path) -> None:
    client = FakeFileClient(b"<html>maintenance</html>")
    with pytest.raises(ValueError, match="did not return a zip"):
        download_archives(client, "Folder", ["2015_01_X.csv"], tmp_path, no_wait(), 0, 0.0)
    assert not archive_path(tmp_path, "Folder", "2015_01_X.csv").exists()


def test_credentials_come_from_the_credential_store(monkeypatch) -> None:
    stored = SimpleNamespace(username="user@example.com", password="secret")
    monkeypatch.setattr(filelib.keyring, "get_credential", lambda service, user: stored)
    assert get_credentials("svc") == ("user@example.com", "secret")


def test_credentials_fall_back_to_environment(monkeypatch) -> None:
    monkeypatch.setattr(filelib.keyring, "get_credential", lambda service, user: None)
    monkeypatch.setenv("ENTSOE_USERNAME", "env-user")
    monkeypatch.setenv("ENTSOE_PWD", "env-pass")
    assert get_credentials("svc") == ("env-user", "env-pass")


def test_missing_credentials_explain_how_to_store_them(monkeypatch) -> None:
    monkeypatch.setattr(filelib.keyring, "get_credential", lambda service, user: None)
    with pytest.raises(CredentialsError, match="watervalues login"):
        get_credentials("svc")
