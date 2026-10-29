"""Tests for parsing File Library exports and fetching them into the cache (synthetic files)."""

import io
import zipfile

import pandas as pd
import pytest

from watervalues.config import AreaSpec, PriceFilterConfig, load_config
from watervalues.data import exports
from watervalues.data.cache import read_dataset
from watervalues.data.fetch import dataset_areas, fetch_filelib, period_bounds

SE = AreaSpec(eic="10YSE-1--------K", area_type="CTA/CTY")
SE1 = AreaSpec(eic="10Y1001A1001A44P", area_type="BZN")


def tsv(rows: list[dict]) -> str:
    return pd.DataFrame(rows).to_csv(sep="\t", index=False)


def write_zip(path, name: str, text: str, bom: bool = True) -> None:
    payload = ("﻿" if bom else "") + text
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(name, payload.encode("utf-8"))


def generation_row(
    time, area, area_type, ptype, mw, resolution="PT60M", update="2025-10-01 10:00:00"
):
    return {
        "DateTime(UTC)": time,
        "ResolutionCode": resolution,
        "AreaCode": area,
        "AreaDisplayName": "x",
        "AreaTypeCode": area_type,
        "AreaMapCode": "x",
        "ProductionType": ptype,
        "ActualGenerationOutput[MW]": mw,
        "ActualConsumption[MW]": "0.0",
        "UpdateTime(UTC)": update,
    }


def test_generation_export_is_filtered_and_normalised(tmp_path) -> None:
    rows = [
        generation_row("2025-06-01 00:00:00", SE.eic, "CTA/CTY", "Hydro Water Reservoir", "8000.0"),
        generation_row("2025-06-01 00:00:00", SE.eic, "CTA/CTY", "Nuclear", "n/e"),
        generation_row("2025-06-01 00:00:00", SE1.eic, "BZN", "Wind Onshore", "500.0"),
        generation_row("2025-06-01 00:00:00", "10YAT-APG------L", "BZN/CTA/CTY", "Biomass", "1.0"),
    ]
    path = tmp_path / "gen.zip"
    write_zip(path, "2025_06_gen.csv", tsv(rows))
    raw = exports.read_export(path, exports.GENERATION_COLUMNS, [SE.eic, SE1.eic], chunk_rows=2)
    assert set(raw["AreaCode"]) == {SE.eic, SE1.eic}  # Austria dropped while reading
    se = exports.parse_generation(exports.select_area(raw, SE))
    assert se["psr_type"].tolist() == ["B12", "B14"]
    assert se["value_mw"].iloc[0] == 8000.0
    assert pd.isna(se["value_mw"].iloc[1])  # "n/e" is missing, not zero
    assert str(se["start_utc"].dtype).startswith("datetime64") and se["start_utc"].dt.tz is not None
    assert se["resolution_min"].tolist() == [60, 60]


def test_select_area_requires_a_single_area_type_when_unset() -> None:
    raw = pd.DataFrame({"AreaCode": [SE.eic, SE.eic], "AreaTypeCode": ["CTA", "CTY"]})
    with pytest.raises(ValueError, match="several AreaTypeCodes"):
        exports.select_area(raw, AreaSpec(eic=SE.eic, area_type=None))
    assert len(exports.select_area(raw, AreaSpec(eic=SE.eic, area_type="CTY"))) == 1


def test_resolution_codes() -> None:
    assert exports.resolution_minutes(pd.Series(["PT15M", "PT60M"])).tolist() == [15, 60]
    with pytest.raises(ValueError, match="P1D"):
        exports.resolution_minutes(pd.Series(["P1D"]))


def price_rows(contract="Day-ahead", sequence="", currency="EUR", value="42.5"):
    return pd.DataFrame(
        {
            "DateTime(UTC)": ["2025-06-01 00:00:00"],
            "ResolutionCode": ["PT60M"],
            "AreaCode": [SE1.eic],
            "AreaTypeCode": ["BZN"],
            "ContractType": [contract],
            "Sequence": [sequence],
            "Price[Currency/MWh]": [value],
            "Currency": [currency],
            "UpdateTime(UTC)": ["2025-06-02 00:00:00"],
        }
    )


FILTER = PriceFilterConfig(contract_type="Day-ahead", currency="EUR", sequences=("", "1"))


def test_prices_keep_only_the_day_ahead_auction() -> None:
    rows = pd.concat(
        [
            price_rows(sequence=" "),  # the exports mark "no sequence" with a single space
            price_rows(sequence="1"),
            price_rows(sequence="2"),  # a second auction, e.g. EXAA
            price_rows(contract="Intraday"),
        ]
    )
    prices = exports.parse_prices(rows, FILTER)
    assert prices["sequence"].tolist() == ["", "1"]
    assert prices["price_eur_mwh"].tolist() == [42.5, 42.5]


def test_prices_in_another_currency_are_rejected() -> None:
    with pytest.raises(ValueError, match="currencies"):
        exports.parse_prices(price_rows(currency="SEK"), FILTER)


def test_reservoir_labels_must_be_iso_weeks() -> None:
    rows = pd.DataFrame(
        {
            "Year": ["2015", "2016"],
            "Week": ["53", "53"],  # 2015 has 53 ISO weeks, 2016 does not
            "AreaCode": [SE.eic] * 2,
            "AreaTypeCode": ["CTA/CTY"] * 2,
            "StoredEnergy[MWh]": ["1", "2"],
            "UpdateTime(UTC)": ["2025-08-08 09:36:41"] * 2,
        }
    )
    with pytest.raises(ValueError):
        exports.parse_reservoir(rows)
    assert exports.parse_reservoir(rows.iloc[:1])["iso_week"].tolist() == [53]


def test_period_bounds_cover_the_first_iso_week_and_references() -> None:
    cfg = load_config()
    assert period_bounds(cfg, "load") == ((2014, 12), (2025, 12))
    assert period_bounds(cfg, "reservoir") == ((2014, 1), (2026, 12))
    assert dataset_areas(cfg.data, "price") == ["SE1", "SE2", "SE3", "SE4", "DE_AT_LU", "DE_LU"]


class ZipServer:
    """Fake File Library serving synthetic load exports."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files

    def list_folder(self, folder: str) -> dict[str, str]:
        return {name: str(i) for i, name in enumerate(self.files)}

    def download_single_file_raw(self, folder: str, filename: str) -> bytes:
        return self.files[filename]


def load_zip(rows: list[dict]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("export.csv", tsv(rows))
    return buffer.getvalue()


def test_fetch_writes_deduplicated_yearly_partitions(tmp_path) -> None:
    cfg = load_config()
    data = cfg.data.model_copy(update={"raw_dir": tmp_path})
    cfg = cfg.model_copy(update={"data": data})

    def row(time, mw, update):
        return {
            "DateTime(UTC)": time,
            "ResolutionCode": "PT60M",
            "AreaCode": SE1.eic,
            "AreaDisplayName": "SE1",
            "AreaTypeCode": "BZN",
            "AreaMapCode": "SE1",
            "TotalLoad[MW]": mw,
            "UpdateTime(UTC)": update,
        }

    server = ZipServer(
        {
            "2014_12_ActualTotalLoad_6.1.A_r3.csv": load_zip(
                [row("2014-12-31 23:00:00", "1000", "2025-01-01 00:00:00")]
            ),
            "2015_01_ActualTotalLoad_6.1.A_r3.csv": load_zip(
                [
                    row("2015-01-01 00:00:00", "1100", "2025-01-01 00:00:00"),
                    row("2015-01-01 00:00:00", "1200", "2025-02-01 00:00:00"),  # revision
                ]
            ),
            "2013_12_ActualTotalLoad_6.1.A_r3.csv": load_zip([]),  # outside the period
        }
    )
    summary = fetch_filelib(cfg, ["load"], client=server, progress=lambda _: None)
    assert summary["load"]["SE1"].rows == 2
    assert summary["load"]["SE1"].revisions.duplicates == 1
    cached = read_dataset(tmp_path, "filelib", "load", "SE1")
    assert cached["value_mw"].tolist() == [1000.0, 1200.0]
    assert sorted(p.name for p in (tmp_path / "filelib" / "load" / "SE1").glob("*.parquet")) == [
        "2014.parquet",
        "2015.parquet",
    ]
