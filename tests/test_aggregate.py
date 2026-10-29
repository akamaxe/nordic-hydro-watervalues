"""Tests for time-based aggregation (synthetic data with hand-computed results)."""

import numpy as np
import pandas as pd
import pytest

from watervalues.data.aggregate import (
    drop_revisions,
    hourly_energy,
    hourly_mean,
    per_stage,
    stitch,
    weekly_energy,
    weekly_generation,
    weekly_load_weighted_price,
    weekly_mean,
)
from watervalues.data.calendar import week_calendar

MIN_COVERAGE = 0.95
CAL_2025 = week_calendar(2025, 2025)
WEEK1 = pd.Timestamp("2024-12-30", tz="UTC")  # start of ISO week 2025-W01


def intervals(start, periods, resolution_min, value, **keys) -> pd.DataFrame:
    """Constant-value intervals starting at ``start``."""
    idx = pd.date_range(start, periods=periods, freq=f"{resolution_min}min")
    return pd.DataFrame(
        {"start_utc": idx, "resolution_min": resolution_min, "value_mw": value, **keys}
    )


def test_mixed_hourly_and_quarter_hourly_week_is_exact() -> None:
    half = pd.Timedelta(hours=84)
    df = pd.concat(
        [
            intervals(WEEK1, 84, 60, 1000.0),
            intervals(WEEK1 + half, 84 * 4, 15, 1000.0),
        ]
    )
    weekly = weekly_energy(hourly_energy(df), CAL_2025, MIN_COVERAGE)
    assert weekly.loc[0, "energy_gwh"] == pytest.approx(168.0)
    assert weekly.loc[0, "coverage"] == pytest.approx(1.0)


def test_overlapping_resolutions_count_each_hour_once() -> None:
    df = pd.concat(
        [intervals(WEEK1, 1, 60, 1000.0), intervals(WEEK1, 4, 15, 2000.0)], ignore_index=True
    )
    hourly = hourly_energy(df)
    assert len(hourly) == 1
    assert hourly.loc[0, "energy_mwh"] == pytest.approx(2000.0)  # the finer resolution wins


def test_partial_hour_keeps_its_coverage() -> None:
    hourly = hourly_energy(intervals(WEEK1, 3, 15, 400.0))
    assert hourly.loc[0, "covered_h"] == pytest.approx(0.75)
    assert hourly.loc[0, "energy_mwh"] == pytest.approx(300.0)


def test_duplicated_intervals_are_rejected() -> None:
    df = pd.concat([intervals(WEEK1, 1, 60, 1.0)] * 2, ignore_index=True)
    with pytest.raises(ValueError, match="drop duplicated"):
        hourly_energy(df)


def test_misaligned_interval_is_rejected() -> None:
    df = intervals(WEEK1 + pd.Timedelta(minutes=5), 1, 15, 1.0)
    with pytest.raises(ValueError, match="aligned"):
        hourly_energy(df)


def test_small_gaps_are_imputed_and_large_gaps_are_nan() -> None:
    full = intervals(WEEK1, 168 * 2, 60, 500.0)
    small_gap = full.drop(index=range(10, 13))  # 3 hours missing in week 1
    large_gap = small_gap.drop(index=range(200, 220))  # 20 hours missing in week 2
    weekly = weekly_energy(hourly_energy(large_gap), CAL_2025, MIN_COVERAGE)
    assert weekly.loc[0, "coverage"] == pytest.approx(165 / 168)
    assert weekly.loc[0, "energy_gwh"] == pytest.approx(84.0)  # 500 MW x 168 h
    assert np.isnan(weekly.loc[1, "energy_gwh"])
    assert weekly.loc[1, "coverage"] == pytest.approx(148 / 168)
    assert np.isnan(weekly.loc[2, "energy_gwh"])  # no data at all
    assert weekly.loc[2, "coverage"] == 0


def test_dst_change_does_not_change_week_length() -> None:
    # Week 2025-W13 contains the switch to summer time in Stockholm (30 March).
    local = pd.date_range("2025-03-24 00:00", "2025-03-31 00:00", freq="h", tz="Europe/Stockholm")
    df = pd.DataFrame({"start_utc": local[:-1], "resolution_min": 60, "value_mw": 1000.0})
    weekly = weekly_energy(hourly_energy(df), CAL_2025, min_coverage=0.9)
    w13 = CAL_2025.index[CAL_2025["iso_week"] == 13][0]
    # 167 local hours exist; in UTC they span Sun 23:00 to Sun 22:00, so W13 has 166 h of data.
    assert weekly.loc[w13, "coverage"] == pytest.approx(166 / 168)
    assert weekly.loc[w13, "energy_gwh"] == pytest.approx(168.0)


def test_drop_revisions_keeps_latest_and_counts_conflicts() -> None:
    df = pd.DataFrame(
        {
            "start_utc": [
                WEEK1,
                WEEK1,
                WEEK1 + pd.Timedelta(hours=1),
                WEEK1 + pd.Timedelta(hours=1),
            ],
            "resolution_min": 60,
            "value_mw": [1.0, 2.0, 5.0, 5.0],
            "update_utc": pd.to_datetime(
                ["2025-01-02", "2025-02-01", "2025-01-02", "2025-01-03"], utc=True
            ),
        }
    )
    kept, stats = drop_revisions(df, ["start_utc", "resolution_min"], ["value_mw"])
    assert kept["value_mw"].tolist() == [2.0, 5.0]
    assert stats.duplicates == 2
    assert stats.conflicts == 1


def test_weekly_mean_is_time_weighted() -> None:
    hourly = pd.concat(
        [
            intervals(WEEK1, 84, 60, 10.0),
            intervals(WEEK1 + pd.Timedelta(hours=84), 84 * 4, 15, 30.0),
        ]
    ).rename(columns={"value_mw": "price_eur_mwh"})
    weekly = weekly_mean(hourly_mean(hourly), CAL_2025, MIN_COVERAGE)
    assert weekly.loc[0, "price_eur_mwh"] == pytest.approx(20.0)


def test_folded_week_sums_energy_and_averages_prices() -> None:
    cal = week_calendar(2020, 2020)
    start = pd.Timestamp("2020-12-21", tz="UTC")  # ISO 2020-W52; W53 follows
    energy = weekly_energy(hourly_energy(intervals(start, 336, 60, 100.0)), cal, MIN_COVERAGE)
    assert energy.loc[51, "energy_gwh"] == pytest.approx(33.6)
    prices = pd.concat(
        [intervals(start, 168, 60, 10.0), intervals(start + pd.Timedelta(days=7), 168, 60, 20.0)]
    ).rename(columns={"value_mw": "price_eur_mwh"})
    mean = weekly_mean(hourly_mean(prices), cal, MIN_COVERAGE)
    assert mean.loc[51, "price_eur_mwh"] == pytest.approx(15.0)


def test_load_weighted_price_matches_hand_calculation() -> None:
    def hourly_price(value):
        df = intervals(WEEK1, 168, 60, value).rename(columns={"value_mw": "price_eur_mwh"})
        return hourly_mean(df)

    prices = {"SE1": hourly_price(10.0), "SE3": hourly_price(40.0)}
    loads = {
        "SE1": hourly_energy(intervals(WEEK1, 168, 60, 1000.0)),
        "SE3": hourly_energy(intervals(WEEK1, 168, 60, 3000.0)),
    }
    weekly = weekly_load_weighted_price(prices, loads, CAL_2025, MIN_COVERAGE)
    assert weekly.loc[0, "price_eur_mwh"] == pytest.approx((10 * 1000 + 40 * 3000) / 4000)


def test_load_weighted_price_skips_hours_with_a_missing_zone() -> None:
    se1_price = hourly_mean(
        intervals(WEEK1, 168, 60, 10.0).rename(columns={"value_mw": "price_eur_mwh"})
    )
    se3_price = hourly_mean(
        intervals(WEEK1, 168, 60, 40.0).rename(columns={"value_mw": "price_eur_mwh"})
    )
    se3_load = intervals(WEEK1, 168, 60, 3000.0)
    se3_load.loc[0, "value_mw"] = np.nan  # one hour without SE3 load
    loads = {
        "SE1": hourly_energy(intervals(WEEK1, 168, 60, 1000.0)),
        "SE3": hourly_energy(se3_load),
    }
    weekly = weekly_load_weighted_price(
        {"SE1": se1_price, "SE3": se3_price}, loads, CAL_2025, MIN_COVERAGE
    )
    assert weekly.loc[0, "coverage"] == pytest.approx(167 / 168)
    assert weekly.loc[0, "price_eur_mwh"] == pytest.approx(32.5)  # never the SE1-only price


def test_se_price_is_bounded_by_load_weighted_not_time_averaged_zone_prices() -> None:
    # Price and load peak together in the expensive zone, as on winter evenings in SE3.
    peak = (np.arange(168) % 24 >= 17).astype(float)

    def hourly_price(value):
        df = intervals(WEEK1, 168, 60, value).rename(columns={"value_mw": "price_eur_mwh"})
        return hourly_mean(df)

    prices = {"SE1": hourly_price(20.0), "SE3": hourly_price(30.0 + 70.0 * peak)}
    loads = {
        "SE1": hourly_energy(intervals(WEEK1, 168, 60, 1000.0)),
        "SE3": hourly_energy(intervals(WEEK1, 168, 60, 3000.0 + 6000.0 * peak)),
    }
    se = weekly_load_weighted_price(prices, loads, CAL_2025, MIN_COVERAGE).loc[0, "price_eur_mwh"]
    time_mean = [
        weekly_mean(p, CAL_2025, MIN_COVERAGE).loc[0, "price_eur_mwh"] for p in prices.values()
    ]
    load_weighted = [
        weekly_load_weighted_price({z: prices[z]}, {z: loads[z]}, CAL_2025, MIN_COVERAGE).loc[
            0, "price_eur_mwh"
        ]
        for z in prices
    ]
    assert se > max(time_mean)
    assert min(load_weighted) <= se <= max(load_weighted)


def test_stitch_respects_the_switch_instant() -> None:
    switch = pd.Timestamp("2018-09-30 22:00", tz="UTC")
    hours = pd.date_range("2018-09-30 20:00", periods=4, freq="h", tz="UTC")
    before = pd.DataFrame({"hour_utc": hours, "price_eur_mwh": 1.0})
    after = pd.DataFrame({"hour_utc": hours, "price_eur_mwh": 2.0})
    out = stitch([(before, None, switch), (after, switch, None)])
    assert out["price_eur_mwh"].tolist() == [1.0, 1.0, 2.0, 2.0]


def test_stitch_rejects_overlaps() -> None:
    hours = pd.date_range("2018-09-30 20:00", periods=2, freq="h", tz="UTC")
    frame = pd.DataFrame({"hour_utc": hours, "price_eur_mwh": 1.0})
    with pytest.raises(ValueError, match="overlap"):
        stitch([(frame, None, None), (frame, None, None)])


KNOWN = ["B10", "B11", "B12", "B14", "B16", "B18", "B19", "B20"]
CATEGORIES = {
    "hydro": ["B10", "B11", "B12"],
    "nuclear": ["B14"],
    "wind": ["B18", "B19"],
    "solar": ["B16"],
}


def test_generation_categories_and_structural_zeros() -> None:
    hydro = intervals(WEEK1, 168 * 3, 60, 1000.0, psr_type="B12")
    nuclear = intervals(WEEK1, 168 * 3, 60, 500.0, psr_type="B14")
    # Solar starts reporting in week 3: weeks 1-2 are structural zeros, not gaps.
    solar = intervals(WEEK1 + pd.Timedelta(days=14), 168, 60, 10.0, psr_type="B16")
    other = intervals(WEEK1, 168 * 3, 60, 20.0, psr_type="B20")
    hourly = hourly_energy(pd.concat([hydro, nuclear, solar, other]), by=("psr_type",))
    weekly = weekly_generation(hourly, CAL_2025, CATEGORIES, KNOWN, MIN_COVERAGE, ["B16"])
    assert weekly.loc[0, "hydro_gwh"] == pytest.approx(168.0)
    assert weekly.loc[0, "nuclear_gwh"] == pytest.approx(84.0)
    assert weekly.loc[0, "solar_gwh"] == 0.0
    assert weekly.loc[2, "solar_gwh"] == pytest.approx(1.68)
    assert weekly.loc[0, "wind_gwh"] == 0.0  # no wind type at all in this area
    assert weekly.loc[0, "other_gwh"] == pytest.approx(3.36)
    # Week 4 has no data for any type: every category is missing, not zero.
    assert weekly.loc[3].isna().all()


def test_gap_inside_a_reporting_period_is_not_a_zero() -> None:
    hydro = intervals(WEEK1, 168 * 3, 60, 1000.0, psr_type="B12")
    nuclear = pd.concat(
        [
            intervals(WEEK1, 168, 60, 500.0, psr_type="B14"),
            intervals(WEEK1 + pd.Timedelta(days=14), 168, 60, 500.0, psr_type="B14"),
        ]
    )
    hourly = hourly_energy(pd.concat([hydro, nuclear]), by=("psr_type",))
    weekly = weekly_generation(hourly, CAL_2025, CATEGORIES, KNOWN, MIN_COVERAGE)
    assert np.isnan(weekly.loc[1, "nuclear_gwh"])
    assert weekly.loc[1, "hydro_gwh"] == pytest.approx(168.0)


def test_only_listed_types_count_as_zero_outside_their_reporting_period() -> None:
    # Nuclear stops one week before the end of the data (say, published late): missing.
    hydro = intervals(WEEK1, 168 * 3, 60, 1000.0, psr_type="B12")
    nuclear = intervals(WEEK1, 168 * 2, 60, 500.0, psr_type="B14")
    hourly = hourly_energy(pd.concat([hydro, nuclear]), by=("psr_type",))
    weekly = weekly_generation(hourly, CAL_2025, CATEGORIES, KNOWN, MIN_COVERAGE)
    assert np.isnan(weekly.loc[2, "nuclear_gwh"])
    listed = weekly_generation(hourly, CAL_2025, CATEGORIES, KNOWN, MIN_COVERAGE, ["B14"])
    assert listed.loc[2, "nuclear_gwh"] == 0.0  # a discontinued type, declared as such


def test_unknown_production_type_is_rejected() -> None:
    hourly = hourly_energy(intervals(WEEK1, 1, 60, 1.0, psr_type="B99"), by=("psr_type",))
    with pytest.raises(ValueError, match="unknown production types"):
        weekly_generation(hourly, CAL_2025, CATEGORIES, KNOWN, MIN_COVERAGE)


def test_per_stage_rescales_only_folded_weeks() -> None:
    weekly = pd.DataFrame(
        {"hours": [168, 336], "hydro_gwh": [100.0, 200.0], "storage_gwh": [5.0, 6.0]}
    )
    out = per_stage(weekly, ["hydro_gwh"])
    assert out["hydro_gwh"].tolist() == [100.0, 100.0]
    assert out["storage_gwh"].tolist() == [5.0, 6.0]
