"""Normalised raw cache: ``<raw_dir>/<source>/<dataset>/<area>/<year>.parquet``.

Every source (File Library now, the REST API later) writes the same normalised frames, so the
weekly build does not depend on where the data came from and two sources can be compared.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def dataset_dir(raw_dir: Path, source: str, dataset: str, area: str) -> Path:
    """Directory holding the yearly partitions of one dataset and area."""
    return raw_dir / source / dataset / area


def write_dataset(
    df: pd.DataFrame,
    raw_dir: Path,
    source: str,
    dataset: str,
    area: str,
    year: pd.Series | np.ndarray,
) -> list[Path]:
    """Replace the cached partitions of one dataset and area with ``df``, split by year.

    Parameters
    ----------
    df : pandas.DataFrame
        Normalised frame (see :mod:`watervalues.data.exports`).
    raw_dir : Path
        Root of the raw cache.
    source, dataset, area : str
        Partition keys, e.g. ``"filelib"``, ``"load"``, ``"SE1"``.
    year : Series or ndarray
        Year of each row (UTC year of the interval start, or ISO year of a weekly label).

    Returns
    -------
    list of Path
        The written parquet files.
    """
    target = dataset_dir(raw_dir, source, dataset, area)
    target.mkdir(parents=True, exist_ok=True)
    for old in target.glob("*.parquet"):
        old.unlink()
    written = []
    for value, part in df.groupby(np.asarray(year), sort=True):
        path = target / f"{int(value)}.parquet"
        tmp = path.with_suffix(".parquet.tmp")
        part.reset_index(drop=True).to_parquet(tmp, index=False)
        tmp.replace(path)
        written.append(path)
    return written


def read_dataset(raw_dir: Path, source: str, dataset: str, area: str) -> pd.DataFrame:
    """Read all cached partitions of one dataset and area.

    Raises
    ------
    FileNotFoundError
        If nothing is cached yet (run ``watervalues fetch``).
    """
    files = sorted(dataset_dir(raw_dir, source, dataset, area).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"no cached {dataset} data for {area} from {source}: run `watervalues fetch`"
        )
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
