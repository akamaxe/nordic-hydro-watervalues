"""ENTSO-E REST API source through entsoe-py (``EntsoePandasClient``).

Each query returns one area; results are normalised to the same frames as the File Library
source and written to ``<raw_dir>/entsoe_api/``, so the weekly build is identical for both.
Requests are made per UTC calendar year (entsoe-py splits them further where the API requires).

entsoe-py 0.8.1 can include the token in DEBUG logs and HTTP error messages, so its logger is
kept at WARNING and every error is re-raised with the token redacted.
"""

from __future__ import annotations

import functools
import itertools
import logging
import os
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from watervalues.config import PROJECT_ROOT, AppConfig, DataConfig
from watervalues.data.aggregate import ALLOWED_RESOLUTIONS_MIN, RevisionStats, drop_revisions
from watervalues.data.cache import partition_path, write_dataset, write_partition
from watervalues.data.fetch import SPECS, AreaResult, dataset_areas, period_bounds
from watervalues.data.filelib import CredentialsError, RateLimiter, call_with_retries
from watervalues.data.schema import psr_code

SOURCE = "entsoe_api"
API_KEY_ENV = "ENTSOE_API_KEY"
_HALF_DAY = pd.Timedelta(hours=12)
WEEK_SPAN = pd.Timedelta(days=7)
_REPORT_TZ = "Europe/Stockholm"  # reservoir weeks are published in Swedish local time
_GENERATION, _CONSUMPTION = "Actual Aggregated", "Actual Consumption"


class ApiClient(Protocol):
    """Queries used here (keyword-only start/end), implemented by :class:`EntsoeApi`."""

    def query_generation(self, country_code, *, start, end) -> pd.DataFrame: ...  # noqa: D102
    def query_load(self, country_code, *, start, end) -> pd.DataFrame: ...  # noqa: D102
    def query_day_ahead_prices(self, country_code, *, start, end) -> pd.Series: ...  # noqa: D102
    def reservoir_xml(self, country_code, *, start, end) -> str: ...  # noqa: D102


class EntsoeApi:
    """Thin wrapper around :class:`entsoe.EntsoePandasClient`.

    Reservoir values are requested as raw XML and parsed by :func:`parse_reservoir_xml`:
    entsoe-py 0.8.1 steps weekly points in fixed UTC intervals and drops the last week of a
    period that starts and ends in different daylight-saving regimes.
    """

    def __init__(self, client) -> None:
        self._client = client

    def query_generation(self, country_code, *, start, end) -> pd.DataFrame:
        """Actual generation per production type (entsoe-py, monthly requests)."""
        return self._client.query_generation(country_code, start=start, end=end)

    def query_load(self, country_code, *, start, end) -> pd.DataFrame:
        """Actual total load (entsoe-py, monthly requests)."""
        return self._client.query_load(country_code, start=start, end=end)

    def query_day_ahead_prices(self, country_code, *, start, end) -> pd.Series:
        """Day-ahead (SDAC) prices (entsoe-py)."""
        return self._client.query_day_ahead_prices(country_code, start=start, end=end)

    def reservoir_xml(self, country_code, *, start, end) -> str:
        """Raw A72 document of weekly reservoir filling (at most one year per request)."""
        from entsoe import EntsoeRawClient

        return EntsoeRawClient.query_aggregate_water_reservoirs_and_hydro_storage(
            self._client, country_code, start=start, end=end
        )


class ApiError(RuntimeError):
    """An API request failed (message redacted)."""


def get_api_key(env_file: Path) -> str:
    """Return the API token from the environment or the project's ``.env`` file.

    Raises
    ------
    CredentialsError
        If no token is configured. The token itself is never printed.
    """
    load_dotenv(env_file, override=False)
    key = os.getenv(API_KEY_ENV, "").strip()
    if not key:
        raise CredentialsError(
            f"No ENTSO-E API token found: put {API_KEY_ENV}=<token> in {env_file.name} "
            "(copy .env.example)."
        )
    return key


def redact(text: str, secret: str) -> str:
    """Replace a secret in a message."""
    return text.replace(secret, "***") if secret else text


def make_api_client(
    api_key: str, retry_count: int, retry_delay_s: float, timeout_s: float
) -> EntsoeApi:
    """Create the API client (entsoe-py underneath) with quiet logging."""
    from entsoe import EntsoePandasClient

    logging.getLogger("entsoe").setLevel(logging.WARNING)
    client = EntsoePandasClient(
        api_key=api_key, retry_count=retry_count, retry_delay=retry_delay_s, timeout=timeout_s
    )
    return EntsoeApi(client)


def infer_resolution_min(index: pd.DatetimeIndex) -> np.ndarray:
    """Interval length of each timestamp [min], from the spacing of a sorted index.

    Each point takes the step to the next one when it is a valid resolution, else the step from
    the previous one (the last point before a gap); isolated points take the most common step.
    """
    if len(index) == 0:
        return np.array([], dtype=np.int64)
    nanoseconds = pd.DatetimeIndex(index).as_unit("ns").asi8  # pandas 3 may store us or s
    steps = np.diff(nanoseconds) / pd.Timedelta(minutes=1).value
    forward = np.append(steps, np.nan)
    backward = np.insert(steps, 0, np.nan)
    allowed = np.isin(forward, ALLOWED_RESOLUTIONS_MIN)
    result = np.where(
        allowed, forward, np.where(np.isin(backward, ALLOWED_RESOLUTIONS_MIN), backward, np.nan)
    )
    if np.isnan(result).any():
        valid = steps[np.isin(steps, ALLOWED_RESOLUTIONS_MIN)]
        if valid.size == 0:
            raise ValueError("cannot infer the time resolution of the series")
        values, counts = np.unique(valid, return_counts=True)
        result = np.where(np.isnan(result), values[np.argmax(counts)], result)
    return result.astype(np.int64)


def _utc_index(obj: pd.Series | pd.DataFrame) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(obj.index)
    if index.tz is None:
        raise ValueError("entsoe-py returned a naive index")
    return index.tz_convert("UTC")


def normalise_generation(df: pd.DataFrame) -> pd.DataFrame:
    """Long frame of generation per PSR type from entsoe-py's wide output.

    entsoe-py returns ``(type, metric)`` MultiIndex columns, or flat type names when only
    'Actual Aggregated' is present; both spellings 'pondage' and 'poundage' are accepted.
    """
    df = df.sort_index()
    index = _utc_index(df)
    resolution = infer_resolution_min(index)
    if isinstance(df.columns, pd.MultiIndex):
        columns = {(name, metric) for name, metric in df.columns}
    else:
        columns = {(name, _GENERATION) for name in df.columns}
        df = df.copy()
        df.columns = pd.MultiIndex.from_tuples([(n, _GENERATION) for n in df.columns])
    parts = []
    for name in sorted({name for name, _ in columns}):
        generation = df[(name, _GENERATION)] if (name, _GENERATION) in columns else np.nan
        consumption = df[(name, _CONSUMPTION)] if (name, _CONSUMPTION) in columns else 0.0
        parts.append(
            pd.DataFrame(
                {
                    "start_utc": index,
                    "resolution_min": resolution,
                    "psr_type": psr_code(name),
                    "value_mw": np.asarray(generation, dtype=float)
                    if np.ndim(generation)
                    else np.full(len(index), generation),
                    "consumption_mw": np.asarray(consumption, dtype=float)
                    if np.ndim(consumption)
                    else np.full(len(index), consumption),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def normalise_series(obj: pd.Series | pd.DataFrame, value_column: str) -> pd.DataFrame:
    """Normalise a single series (load or price) to ``start_utc``, ``resolution_min``, value."""
    series = obj.iloc[:, 0] if isinstance(obj, pd.DataFrame) else obj
    series = series.sort_index()
    index = _utc_index(series)
    return pd.DataFrame(
        {
            "start_utc": index,
            "resolution_min": infer_resolution_min(index),
            value_column: series.to_numpy(dtype=float),
        }
    )


def _local_monday(instant: pd.Timestamp) -> date:
    """Local calendar Monday of a week-start instant (robust to a one-hour DST offset)."""
    day = (instant.tz_convert(_REPORT_TZ) + _HALF_DAY).normalize()
    if day.day_of_week != 0:
        raise ValueError(f"weekly period does not start on a Monday: {instant}")
    return day.date()


def parse_reservoir_xml(xml_text: str) -> pd.DataFrame:
    """Weekly reservoir values of an A72 (16.1.D) API document, labelled by ISO week.

    Weeks are counted in calendar days from each period's first local Monday, so a period that
    crosses a daylight-saving change keeps all its weeks. With curve type A03, positions whose
    value repeats are omitted by the sender; they are forward-filled within the period.

    Returns
    -------
    pandas.DataFrame
        ``iso_year``, ``iso_week`` and ``stored_mwh`` [MWh].
    """
    rows = []
    for series in ET.fromstring(xml_text).iterfind(".//{*}TimeSeries"):
        curve = series.findtext("{*}curveType", default="A01")
        for period in series.iterfind("{*}Period"):
            if period.findtext("{*}resolution") != "P7D":
                raise ValueError("reservoir periods must have weekly (P7D) resolution")
            start = pd.Timestamp(period.findtext("{*}timeInterval/{*}start"))
            end = pd.Timestamp(period.findtext("{*}timeInterval/{*}end"))
            weeks = round((end - start) / WEEK_SPAN)  # 7 days, give or take a DST hour
            values = {
                int(point.findtext("{*}position")): float(point.findtext("{*}quantity"))
                for point in period.iterfind("{*}Point")
            }
            monday = _local_monday(start)
            current = None
            for position in range(1, weeks + 1):
                if position in values:
                    current = values[position]
                elif curve != "A03" or current is None:
                    continue
                iso = (monday + timedelta(weeks=position - 1)).isocalendar()
                rows.append((iso.year, iso.week, current))
    return pd.DataFrame(rows, columns=["iso_year", "iso_week", "stored_mwh"])


def api_areas(data: DataConfig, dataset: str) -> list[str]:
    """Areas queried per dataset: the weekly build needs generation for the country only."""
    if dataset == "generation":
        return [data.country]
    return dataset_areas(data, dataset)


def _windows(cfg: AppConfig, dataset: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Request windows in UTC: calendar years covering the needed period."""
    (first_year, first_month), (last_year, _) = period_bounds(cfg, dataset)
    start = pd.Timestamp(year=first_year, month=first_month, day=1, tz="UTC")
    if dataset != "reservoir":
        # The first ISO week of the period starts on the last Monday of December.
        start = pd.Timestamp(date.fromisocalendar(first_year + 1, 1, 1), tz="UTC")
    end = pd.Timestamp(year=last_year + 1, month=1, day=1, tz="UTC")
    edges = [start, *pd.date_range(start, end, freq="YS", inclusive="neither"), end]
    return list(itertools.pairwise(edges))


def _query(
    client: ApiClient, dataset: str, eic: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """One API request, normalised (raises entsoe's NoMatchingDataError when empty)."""
    if dataset == "generation":
        return normalise_generation(client.query_generation(eic, start=start, end=end))
    if dataset == "load":
        return normalise_series(client.query_load(eic, start=start, end=end), "value_mw")
    if dataset == "price":
        prices = client.query_day_ahead_prices(eic, start=start, end=end)
        return normalise_series(prices, "price_eur_mwh").assign(sequence="")
    return parse_reservoir_xml(client.reservoir_xml(eic, start=start, end=end))


def fetch_api(
    cfg: AppConfig,
    datasets: Sequence[str],
    *,
    client: ApiClient | None = None,
    api_key: str = "",
    force: bool = False,
    progress: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, dict[str, AreaResult]]:
    """Query the ENTSO-E API into the ``entsoe_api`` raw cache, resuming where it stopped.

    Interval datasets are requested and cached per UTC year, so an interrupted run skips the
    years already cached. Timeouts, connection errors and server errors are retried with
    exponential backoff.

    Parameters
    ----------
    cfg : AppConfig
        Model configuration.
    datasets : sequence of str
        Any of ``generation``, ``load``, ``price`` and ``reservoir``.
    client : ApiClient or None
        API client; ``None`` creates one with the token from ``.env``.
    api_key : str
        Token used only to redact error messages when ``client`` is given.
    force : bool
        Request again the years already cached.
    progress : callable
        Receives progress messages.
    sleep : callable
        Pause between retries (injected in tests).

    Returns
    -------
    dict
        Per dataset and area, the number of rows cached and the duplicates removed.
    """
    from entsoe.exceptions import NoMatchingDataError

    data = cfg.data
    settings = data.api
    raw_dir = PROJECT_ROOT / data.raw_dir
    if client is None:
        api_key = get_api_key(PROJECT_ROOT / settings.env_file)
        client = make_api_client(
            api_key, settings.retry_count, settings.retry_delay_s, settings.timeout_s
        )
    limiter = RateLimiter(0.0)

    def request(dataset: str, area: str, start: pd.Timestamp, end: pd.Timestamp):
        try:
            return call_with_retries(
                functools.partial(_query, client, dataset, data.areas[area].eic, start, end),
                limiter,
                settings.max_retries,
                settings.backoff_s,
                sleep,
            )
        except NoMatchingDataError:
            return None
        except Exception as exc:  # re-raise without the token in the message
            raise ApiError(
                f"{dataset}/{area} {start:%Y-%m-%d}: "
                f"{redact(f'{type(exc).__name__}: {exc}', api_key)}"
            ) from None

    summary: dict[str, dict[str, AreaResult]] = {}
    for dataset in datasets:
        spec = SPECS[dataset]
        summary[dataset] = {}
        for area in api_areas(data, dataset):
            if dataset == "reservoir":
                parts = [request(dataset, area, s, e) for s, e in _windows(cfg, dataset)]
                parts = [p for p in parts if p is not None]
                if not parts:
                    progress(f"warning: no {dataset} data for {area}")
                    continue
                frame, stats = drop_revisions(pd.concat(parts), spec.keys, spec.values)
                write_dataset(frame, raw_dir, SOURCE, dataset, area, frame["iso_year"])
                summary[dataset][area] = AreaResult(rows=len(frame), revisions=stats)
                progress(f"{dataset}/{area}: {len(frame):,} weeks cached")
                continue
            rows = 0
            for start, end in _windows(cfg, dataset):
                path = partition_path(raw_dir, SOURCE, dataset, area, start.year)
                if path.exists() and not force:
                    rows += len(pd.read_parquet(path, columns=["start_utc"]))
                    continue
                frame = request(dataset, area, start, end)
                if frame is None:
                    continue
                frame = frame[(frame["start_utc"] >= start) & (frame["start_utc"] < end)]
                frame, _ = drop_revisions(frame, spec.keys, spec.values)
                write_partition(frame, raw_dir, SOURCE, dataset, area, start.year)
                rows += len(frame)
                progress(f"{dataset}/{area} {start:%Y}: {len(frame):,} rows")
            if rows == 0:
                progress(f"warning: no {dataset} data for {area}")
                continue
            summary[dataset][area] = AreaResult(rows=rows, revisions=RevisionStats(0, 0))
            progress(f"{dataset}/{area}: {rows:,} rows cached")
    return summary


def check_api(cfg: AppConfig, client: ApiClient | None = None, api_key: str = "") -> int:
    """Make one small request (one day of prices) to verify the token.

    Returns
    -------
    int
        Number of prices received.

    Raises
    ------
    ApiError
        With the token redacted, if the request fails.
    """
    settings = cfg.data.api
    if client is None:
        api_key = get_api_key(PROJECT_ROOT / settings.env_file)
        client = make_api_client(
            api_key, 1, settings.retry_delay_s, settings.timeout_s
        )  # one attempt
    zone = cfg.data.areas[cfg.data.price_zones[0]].eic
    start = pd.Timestamp(year=cfg.period.last_iso_year, month=1, day=1, tz="UTC")
    try:
        prices = client.query_day_ahead_prices(zone, start=start, end=start + pd.Timedelta(days=1))
    except Exception as exc:
        raise ApiError(redact(f"{type(exc).__name__}: {exc}", api_key)) from None
    return len(prices)
