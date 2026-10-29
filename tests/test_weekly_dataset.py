"""Checks on the committed weekly dataset (data/processed/weekly_se.csv)."""

import numpy as np
import pytest

from watervalues.config import PROJECT_ROOT, load_config
from watervalues.data.build import load_known_gaps, load_weekly, unexplained_missing
from watervalues.units import HOURS_PER_WEEK, MODEL_WEEKS_PER_YEAR

CFG = load_config()


@pytest.fixture(scope="module")
def weekly():
    return load_weekly(PROJECT_ROOT / CFG.data.weekly_csv)


def test_one_row_per_model_week(weekly) -> None:
    years = CFG.period.last_iso_year - CFG.period.first_iso_year + 1
    assert len(weekly) == years * MODEL_WEEKS_PER_YEAR
    assert (weekly.groupby("iso_year").size() == MODEL_WEEKS_PER_YEAR).all()
    folded = weekly[weekly["hours"] != HOURS_PER_WEEK]
    assert folded["iso_year"].tolist() == [2015, 2020]  # the ISO years with a week 53
    assert (folded["iso_week"] == MODEL_WEEKS_PER_YEAR).all()
    assert (folded["hours"] == 2 * HOURS_PER_WEEK).all()


def test_storage_is_continuous_and_inside_the_capacity(weekly) -> None:
    np.testing.assert_allclose(weekly["storage_end_gwh"].iloc[:-1], weekly["storage_gwh"].iloc[1:])
    storage = weekly["storage_gwh"]
    assert storage.notna().all()
    assert (storage > 0).all()
    assert (storage < CFG.data.sanity.storage_capacity_ref_gwh).all()


def test_every_missing_value_is_documented(weekly) -> None:
    gaps = load_known_gaps(PROJECT_ROOT / CFG.data.known_gaps_file)
    assert unexplained_missing(weekly, gaps) == []
