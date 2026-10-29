"""Tests for the reservoir labelling convention (synthetic reservoir with known inflow)."""

import numpy as np
import pandas as pd
import pytest

from watervalues.data.calendar import assign_bins, week_calendar
from watervalues.data.reservoir import (
    LabelSemantics,
    ReferencePoint,
    boundary_snapshots,
    diagnose_alignment,
    fill_short_gaps,
    labelled_series,
    reference_residuals,
    shift_ratio,
    storage_columns,
)

WINTER = (2, 14)
CALM_DH_GWH, VOLATILE_DH_GWH = 115, 350  # std of weekly production changes: 2015-21 vs 2022-25


def synthetic_reservoir(seed: int = 0, years: int = 4):
    """Storage, smooth inflow and noisy production for consecutive ISO weeks from 2015-W01."""
    rng = np.random.default_rng(seed)
    starts = pd.date_range("2014-12-29", periods=52 * years + 1, freq="7D", tz="UTC")
    n = len(starts)
    week = np.asarray(starts.isocalendar()["week"])
    inflow = 300 + 2500 * np.exp(-(((week - 24) / 4) ** 2)) + rng.normal(0, 10, n)  # [GWh]
    production = 1500 + rng.normal(0, 200, n)  # [GWh], follows demand
    storage = 20_000 + np.concatenate([[0.0], np.cumsum(inflow - production)])  # boundaries
    return starts, inflow, production, storage


def noisy_end_of_week(seed: int, dh_std_gwh: float, noise_gwh: float, years: int = 11):
    """End-of-week labels with reporting noise; production anomalies persist from week to week."""
    rng = np.random.default_rng(seed)
    starts = pd.date_range("2014-12-29", periods=52 * years + 1, freq="7D", tz="UTC")
    n = len(starts)
    season = np.cos(2 * np.pi * np.arange(n) / 52)
    phi = 0.6  # AR(1) persistence; innovations scaled so that std(diff) = dh_std_gwh
    anomaly = np.zeros(n)
    for i in range(1, n):
        anomaly[i] = phi * anomaly[i - 1] + rng.normal(0, dh_std_gwh * np.sqrt((1 + phi) / 2))
    production = 1300 + 300 * season + anomaly
    inflow = 1300 - 900 * season + rng.normal(0, 40, n)
    end_storage = 20_000 + np.cumsum(inflow - production) + rng.normal(0, noise_gwh, n)
    return pd.Series(end_storage, index=starts), pd.Series(production, index=starts)


def publish(starts, storage, semantics: LabelSemantics) -> pd.DataFrame:
    """Weekly labels as a data provider would publish them under ``semantics``."""
    start_of_week, end_of_week = storage[:-1], storage[1:]
    values = {
        LabelSemantics.START_OF_WEEK: start_of_week,
        LabelSemantics.END_OF_WEEK: end_of_week,
        LabelSemantics.WEEKLY_MEAN: (start_of_week + end_of_week) / 2,
    }[semantics]
    iso = starts.isocalendar()
    return pd.DataFrame(
        {
            "iso_year": np.asarray(iso["year"]),
            "iso_week": np.asarray(iso["week"]),
            "stored_mwh": values * 1000.0,
        }
    )


@pytest.mark.parametrize("truth", list(LabelSemantics))
def test_diagnosis_recovers_the_true_convention(truth: LabelSemantics) -> None:
    starts, _, production, storage = synthetic_reservoir()
    published = labelled_series(publish(starts, storage, truth))
    hydro = pd.Series(production, index=starts)
    evidence = diagnose_alignment(published, hydro, WINTER)
    assert evidence.best is truth
    assert set(evidence.best_by_year.values()) == {truth.value}
    if truth is LabelSemantics.WEEKLY_MEAN:
        # A weekly mean is only weakly separated from end-of-week values by roughness alone;
        # the external reference is the decisive test for it.
        assert evidence.ratio < 1.0
    else:
        assert evidence.ratio < 0.2


def test_shift_ratio_compares_a_snapshot_convention_with_its_one_week_shift() -> None:
    starts, _, production, storage = synthetic_reservoir()
    published = labelled_series(publish(starts, storage, LabelSemantics.END_OF_WEEK))
    evidence = diagnose_alignment(published, pd.Series(production, index=starts), WINTER)
    assert shift_ratio(evidence, LabelSemantics.END_OF_WEEK) < 0.2
    assert shift_ratio(evidence, LabelSemantics.START_OF_WEEK) > 5
    with pytest.raises(ValueError, match="no one-week shift"):
        shift_ratio(evidence, LabelSemantics.WEEKLY_MEAN)


@pytest.mark.parametrize("seed", range(5))
def test_reporting_noise_favours_the_weekly_mean_but_not_a_one_week_shift(seed: int) -> None:
    # End-of-week truth with 60 GWh of storage reporting noise and calm production (2015-21):
    # the centred difference of the weekly-mean estimator smooths the noise, so roughness
    # alone picks the wrong convention, while the one-week shift is still clearly worse.
    published, hydro = noisy_end_of_week(seed, CALM_DH_GWH, noise_gwh=60)
    evidence = diagnose_alignment(published, hydro, WINTER)
    assert evidence.best is LabelSemantics.WEEKLY_MEAN
    assert shift_ratio(evidence, LabelSemantics.END_OF_WEEK) < 1


@pytest.mark.parametrize("seed", range(5))
def test_volatile_production_lets_roughness_find_end_of_week_despite_noise(seed: int) -> None:
    published, hydro = noisy_end_of_week(seed, VOLATILE_DH_GWH, noise_gwh=60)
    evidence = diagnose_alignment(published, hydro, WINTER)
    assert evidence.best is LabelSemantics.END_OF_WEEK
    assert shift_ratio(evidence, LabelSemantics.END_OF_WEEK) < 0.5


def test_end_of_week_storage_closes_the_balance_exactly() -> None:
    starts, inflow, production, storage = synthetic_reservoir()
    labels = publish(starts, storage, LabelSemantics.END_OF_WEEK)
    # The start of the first week is the end of the week before: 2014-W52.
    prior = pd.DataFrame({"iso_year": [2014], "iso_week": [52], "stored_mwh": [storage[0] * 1e3]})
    labels = pd.concat([prior, labels], ignore_index=True)
    cal = week_calendar(2015, 2018)  # 2015 has 53 ISO weeks: its week 52 is folded
    columns = storage_columns(boundary_snapshots(labels, LabelSemantics.END_OF_WEEK), cal)
    bins = assign_bins(starts, cal)
    inside = bins >= 0
    hydro_bin = np.bincount(bins[inside], weights=production[inside], minlength=len(cal))
    inflow_bin = np.bincount(bins[inside], weights=inflow[inside], minlength=len(cal))
    balance = columns["storage_end_gwh"] - columns["storage_gwh"] + hydro_bin
    np.testing.assert_allclose(balance, inflow_bin, atol=1e-6)
    assert columns.notna().all().all()


def test_start_of_week_storage_uses_the_next_label_as_end() -> None:
    starts, _, _, storage = synthetic_reservoir(years=1)
    labels = publish(starts, storage, LabelSemantics.START_OF_WEEK)
    cal = week_calendar(2015, 2015)
    columns = storage_columns(boundary_snapshots(labels, LabelSemantics.START_OF_WEEK), cal)
    np.testing.assert_allclose(columns["storage_gwh"].iloc[0], storage[0])
    np.testing.assert_allclose(columns["storage_end_gwh"].iloc[0], storage[1])


def test_folded_week_ends_at_the_next_iso_year() -> None:
    starts, _, _, storage = synthetic_reservoir(years=2)
    labels = publish(starts, storage, LabelSemantics.END_OF_WEEK)
    cal = week_calendar(2015, 2015)
    columns = storage_columns(boundary_snapshots(labels, LabelSemantics.END_OF_WEEK), cal)
    # 2015-W52 is folded with W53: it starts after W51 and ends after W53 (the 53rd boundary).
    np.testing.assert_allclose(columns["storage_gwh"].iloc[51], storage[51])
    np.testing.assert_allclose(columns["storage_end_gwh"].iloc[51], storage[53])


def test_only_short_gaps_are_interpolated() -> None:
    index = pd.date_range("2016-01-04", periods=8, freq="7D", tz="UTC")
    values = pd.Series([10.0, np.nan, 30.0, np.nan, np.nan, 60.0, 70.0, 80.0], index=index)
    filled, interpolated = fill_short_gaps(values.dropna(), max_weeks=1)
    assert filled.iloc[1] == pytest.approx(20.0)
    assert np.isnan(filled.iloc[3]) and np.isnan(filled.iloc[4])  # two missing weeks: kept
    assert interpolated == [index[1]]


def test_weekly_mean_semantics_is_refused() -> None:
    starts, _, _, storage = synthetic_reservoir(years=1)
    labels = publish(starts, storage, LabelSemantics.WEEKLY_MEAN)
    with pytest.raises(ValueError, match="inflow formulation"):
        boundary_snapshots(labels, LabelSemantics.WEEKLY_MEAN)


def test_duplicated_labels_are_rejected() -> None:
    starts, _, _, storage = synthetic_reservoir(years=1)
    labels = publish(starts, storage, LabelSemantics.END_OF_WEEK)
    with pytest.raises(ValueError, match="duplicated"):
        labelled_series(pd.concat([labels, labels.iloc[:1]]))


def test_reference_residuals_identify_end_of_week_values() -> None:
    starts, _, _, storage = synthetic_reservoir(years=1)
    published = labelled_series(publish(starts, storage, LabelSemantics.END_OF_WEEK))
    # External report: storage at the end of ISO week 2015-W10 equals the boundary after it.
    reference = ReferencePoint(area="SE", iso_year=2015, iso_week=10, stored_gwh=storage[10])
    residuals = reference_residuals({"SE": published}, [reference])
    assert residuals.loc[0, "end_of_week"] == pytest.approx(0.0)
    assert abs(residuals.loc[0, "start_of_week"]) > 1.0
