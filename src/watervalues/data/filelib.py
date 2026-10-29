"""ENTSO-E File Library source: authenticated download of the bulk exports.

The File Library is accessed with the Transparency Platform website login (no REST API token)
through :class:`entsoe.files.EntsoeFileClient` from entsoe-py. Each export covers one month (one
year for reservoirs) and every area. Downloads are kept as zips in ``<raw_dir>/filelib/_archives``
with a manifest, so a fetch can be resumed and the raw cache rebuilt without downloading again.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import time
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from typing import Protocol

import keyring
import requests

USERNAME_ENV = "ENTSOE_USERNAME"  # same environment variables as entsoe-py
PASSWORD_ENV = "ENTSOE_PWD"
_FILE_NAME = re.compile(r"^(?P<year>\d{4})_(?:(?P<month>\d{2})_)?(?P<rest>.+)\.csv$")


class FileClient(Protocol):
    """The part of :class:`entsoe.files.EntsoeFileClient` used here."""

    def list_folder(self, folder: str) -> dict[str, str]:
        """Return ``{file name: file id}`` for a File Library folder."""
        ...

    def download_single_file_raw(self, folder: str, filename: str) -> bytes:
        """Return one export as zip bytes."""
        ...


class CredentialsError(RuntimeError):
    """No Transparency Platform website login is available."""


class RateLimitedError(RuntimeError):
    """The File Library answered HTTP 429; retrying now would extend the ban."""


def get_credentials(service: str) -> tuple[str, str]:
    """Return the website login from the OS credential store, else from the environment.

    Parameters
    ----------
    service : str
        Credential-store service name (``keyring``), e.g. ``"nordic-hydro-watervalues/entsoe-tp"``.

    Returns
    -------
    tuple of str
        ``(username, password)``. Never print or log them.
    """
    credential = keyring.get_credential(service, None)
    if credential is not None and credential.username and credential.password:
        return credential.username, credential.password
    username, password = os.getenv(USERNAME_ENV), os.getenv(PASSWORD_ENV)
    if username and password:
        return username, password
    raise CredentialsError(
        "No ENTSO-E Transparency Platform login found. Store it once (it is never shown) with:\n"
        f"  python -m keyring set {service} <your-website-email>\n"
        f"or set {USERNAME_ENV} and {PASSWORD_ENV}."
    )


def make_client(service: str, timeout_s: float) -> FileClient:
    """Create an authenticated File Library client.

    Parameters
    ----------
    service : str
        Credential-store service name holding the website login.
    timeout_s : float
        Timeout of every HTTP request [s].

    Returns
    -------
    FileClient
        A logged-in :class:`entsoe.files.EntsoeFileClient`.
    """
    from entsoe.files import EntsoeFileClient  # imported lazily: it logs in on creation

    username, password = get_credentials(service)
    session = requests.Session()
    # EntsoeFileClient does not pass a timeout when downloading; give every request one.
    session.request = functools.partial(session.request, timeout=timeout_s)  # type: ignore[method-assign]
    return EntsoeFileClient(username=username, pwd=password, session=session, timeout=timeout_s)


@dataclass
class RateLimiter:
    """Space calls at least ``min_interval_s`` apart."""

    min_interval_s: float
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    _last: float | None = field(default=None, init=False)

    def wait(self) -> None:
        """Block until the next call is allowed."""
        now = self.clock()
        if self._last is not None:
            delay = self._last + self.min_interval_s - now
            if delay > 0:
                self.sleep(delay)
                now += delay
        self._last = now


def call_with_retries[T](
    func: Callable[[], T],
    limiter: RateLimiter,
    max_retries: int,
    backoff_s: float,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``func`` politely: rate-limited, with backoff on transient errors.

    Server errors (5xx), connection errors and timeouts are retried with exponential backoff.
    HTTP 429 is never retried, because the File Library then extends a ten-minute ban.

    Raises
    ------
    RateLimitedError
        On HTTP 429.
    """
    for attempt in range(max_retries + 1):
        limiter.wait()
        try:
            return func()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == HTTPStatus.TOO_MANY_REQUESTS:
                raise RateLimitedError(
                    "File Library rate limit reached (HTTP 429): wait ten minutes and resume"
                ) from exc
            retryable = status is not None and status >= HTTPStatus.INTERNAL_SERVER_ERROR
            if not retryable or attempt == max_retries:
                raise
        except (requests.ConnectionError, requests.Timeout):
            if attempt == max_retries:
                raise
        sleep(backoff_s * 2**attempt)
    raise AssertionError("unreachable")  # pragma: no cover


def file_period(name: str) -> tuple[int, int | None]:
    """Return ``(year, month)`` of an export file name; ``month`` is None for yearly files.

    Examples: ``2018_08_ActualTotalLoad_6.1.A_r3.csv`` -> ``(2018, 8)``.
    """
    match = _FILE_NAME.match(name)
    if match is None:
        raise ValueError(f"unexpected File Library file name: {name!r}")
    month = match["month"]
    return int(match["year"]), None if month is None else int(month)


def select_files(names: Iterable[str], first: tuple[int, int], last: tuple[int, int]) -> list[str]:
    """Pick the export files whose period lies within ``first``..``last`` (year, month).

    Yearly files match when their year lies within the range.
    """
    selected = []
    for name in names:
        try:
            year, month = file_period(name)
        except ValueError:
            continue  # logs and readme files in the folders
        keep = first[0] <= year <= last[0] if month is None else first <= (year, month) <= last
        if keep:
            selected.append(name)
    return sorted(selected)


def archive_path(raw_dir: Path, folder: str, filename: str) -> Path:
    """Local path of a downloaded export (zip)."""
    return raw_dir / "filelib" / "_archives" / folder / f"{filename}.zip"


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(path)


@dataclass
class Manifest:
    """Record of downloaded exports (``<raw_dir>/filelib/_manifest.json``)."""

    path: Path
    entries: dict[str, dict[str, str]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> Manifest:
        """Read the manifest, or start an empty one."""
        if path.exists():
            return cls(path, json.loads(path.read_text(encoding="utf-8")))
        return cls(path)

    def record(self, folder: str, filename: str, content: bytes) -> None:
        """Add or replace the entry of one download and save the manifest."""
        self.entries[f"{folder}/{filename}"] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "downloaded_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        text = json.dumps(self.entries, indent=1, sort_keys=True)
        _atomic_write(self.path, text.encode("utf-8"))

    def digest(self) -> str:
        """Fingerprint of all downloads (sha256 over the sorted entries)."""
        joined = "\n".join(f"{k}={v['sha256']}" for k, v in sorted(self.entries.items()))
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def download_archives(
    client: FileClient,
    folder: str,
    filenames: Iterable[str],
    raw_dir: Path,
    limiter: RateLimiter,
    max_retries: int,
    backoff_s: float,
    *,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> list[Path]:
    """Download exports that are not cached yet (all of them with ``force``).

    Returns
    -------
    list of Path
        Local zip paths, in the order of ``filenames``.
    """
    manifest = Manifest.load(raw_dir / "filelib" / "_manifest.json")
    paths = []
    for name in filenames:
        path = archive_path(raw_dir, folder, name)
        if force or not path.exists():
            content = call_with_retries(
                functools.partial(client.download_single_file_raw, folder, name),
                limiter,
                max_retries,
                backoff_s,
            )
            if not zipfile.is_zipfile(BytesIO(content)):
                raise ValueError(f"File Library did not return a zip for {folder}/{name}")
            _atomic_write(path, content)
            manifest.record(folder, name, content)
            if progress is not None:
                progress(f"downloaded {folder}/{name} ({len(content) / 1e6:.1f} MB)")
        paths.append(path)
    return paths
