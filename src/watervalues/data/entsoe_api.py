"""ENTSO-E REST API source through entsoe-py (``EntsoePandasClient``).

Each query returns one area; results are normalised to the same frames as the File Library
source and written to ``<raw_dir>/entsoe_api/``, so the weekly build is identical for both.
Requests are made per UTC calendar year (entsoe-py splits them further where the API requires).

entsoe-py 0.8.1 can include the token in DEBUG logs and HTTP error messages, so its logger is
kept at WARNING and every error is re-raised with the token redacted.
"""

from __future__ import annotations

import itertools
import logging
import os
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from watervalues.config import PROJECT_ROOT, AppConfig, DataConfig
from watervalues.data.aggregate import ALLOWED_RESOLUTIONS_MIN, drop_revisions
from watervalues.data.cache import write_dataset
from watervalues.data.fetch import SPECS, AreaResult, dataset_areas, period_bounds
from watervalues.data.filelib import CredentialsError
from watervalues.data.schema import psr_code

SOURCE = "entsoe_api"
API_KEY_ENV = "ENTSOE_API_KEY"
_HALF_DAY = pd.Timedelta(hours=12)
_REPORT_TZ = "Europe/Stockholm"  # reservoir weeks are published in Swedish local time
_GENERATION, _CONSUMPTION = "Actual Aggregated", "Actual Consumption"


class ApiClient(Protocol):
    """The part of :class:`entsoe.EntsoePandasClient` used here (keyword-only start/end)."""

    def query_generation(self, country_code, *, start, end, **kwargs) -> pd.DataFrame: ...  # noqa: D102
    def query_load(self, country_code, *, start, end) -> pd.DataFrame: ...  # noqa: D102
    def query_day_ahead_prices(self, country_code, *, start, end) -> pd.Series: ...  # noqa: D102
    def query_aggregate_water_reservoirs_and_hydro_storage(  # noqa: D102
        self, country_code, *, start, end
    ) -> pd.Series: ...


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


def make_api_client(api_key: str, retry_count: int, retry_delay_s: float, timeout_s: float):
    """Create an :class:`entsoe.EntsoePandasClient` with quiet logging."""
    from entsoe import EntsoePandasClient

    logging.getLogger("entsoe").setLevel(logging.WARNING)
    return EntsoePandasClient(
        api_key=api_key, retry_count=retry_count, retry_delay=retry_delay_s, timeout=timeout_s
    )


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


def normalise_reservoir(series: pd.Series) -> pd.DataFrame:
    """Map weekly reservoir values to ISO week labels.

    Each week starts on Monday 00:00 Swedish time; entsoe-py steps fixed 7-day intervals in UTC,
    which can drift by an hour across daylight-saving changes, so the label is taken from the
    nearest local day.
    """
    local = _utc_index(series).tz_convert(_REPORT_TZ)
    days = (local + _HALF_DAY).normalize()
    if (days.day_of_week != 0).any():
        raise ValueError("reservoir values do not start on Mondays")
    iso = [date(d.year, d.month, d.day).isocalendar() for d in days]
    return pd.DataFrame(
        {
            "iso_year": [i.year for i in iso],
            "iso_week": [i.week for i in iso],
            "stored_mwh": series.to_numpy(dtype=float),
        }
    )


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


def fetch_api(
    cfg: AppConfig,
    datasets: Sequence[str],
    *,
    client: ApiClient | None = None,
    api_key: str = "",
    progress: Callable[[str], None] = print,
) -> dict[str, dict[str, AreaResult]]:
    """Query the ENTSO-E API and rebuild the ``entsoe_api`` raw cache.

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
    progress : callable
        Receives progress messages.

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
    overlap = pd.Timedelta(days=settings.reservoir_overlap_days)
    summary: dict[str, dict[str, AreaResult]] = {}
    for dataset in datasets:
        summary[dataset] = {}
        for area in api_areas(data, dataset):
            eic = data.areas[area].eic
            parts = []
            for start, end in _windows(cfg, dataset):
                try:
                    if dataset == "generation":
                        frame = normalise_generation(
                            client.query_generation(eic, start=start, end=end)
                        )
                    elif dataset == "load":
                        frame = normalise_series(
                            client.query_load(eic, start=start, end=end), "value_mw"
                        )
                    elif dataset == "price":
                        frame = normalise_series(
                            client.query_day_ahead_prices(eic, start=start, end=end),
                            "price_eur_mwh",
                        ).assign(sequence="")
                    else:  # overlapping windows: entsoe-py 0.8.1 drops year-boundary weeks
                        frame = normalise_reservoir(
                            client.query_aggregate_water_reservoirs_and_hydro_storage(
                                eic, start=start - overlap, end=end + overlap
                            )
                        )
                except NoMatchingDataError:
                    continue
                except Exception as exc:  # re-raise without the token in the message
                    raise ApiError(
                        f"{dataset}/{area} {start:%Y-%m-%d}: "
                        f"{redact(f'{type(exc).__name__}: {exc}', api_key)}"
                    ) from None
                if "start_utc" in frame:
                    frame = frame[(frame["start_utc"] >= start) & (frame["start_utc"] < end)]
                parts.append(frame)
                progress(f"{dataset}/{area} {start:%Y}: {len(frame):,} rows")
            if not parts:
                progress(f"warning: no {dataset} data for {area}")
                continue
            spec = SPECS[dataset]
            frame, stats = drop_revisions(
                pd.concat(parts, ignore_index=True), spec.keys, spec.values
            )
            years = frame["iso_year"] if dataset == "reservoir" else frame["start_utc"].dt.year
            write_dataset(frame, raw_dir, SOURCE, dataset, area, years)
            summary[dataset][area] = AreaResult(rows=len(frame), revisions=stats)
            progress(
                f"{dataset}/{area}: {len(frame):,} rows cached ({stats.duplicates} duplicates)"
            )
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
