# Data

Provenance, licence, processing rules and known gaps of the data behind the model.

| Path | Content | Tracked by git |
|---|---|---|
| `raw/<source>/<dataset>/<area>/<year>.parquet` | Normalised raw cache, rebuilt by `watervalues fetch` | no |
| `raw/filelib/_archives/`, `raw/filelib/_manifest.json` | File Library downloads and their checksums | no |
| `external/` | Reference files, e.g. `energiforetagen/` downloaded by `watervalues validate` | no (except its README) |
| `processed/weekly_se.csv` | Weekly dataset used by the model | yes |
| `processed/build_report.txt` | Sanity report of the build that produced the CSV | yes |
| `processed/validation_energiforetagen.txt` | Comparison of the ENTSO-E series with Energiföretagen's weekly statistics | yes |
| `known_gaps.yaml` | Documented missing values of the weekly dataset | yes |

## Sources

All series come from the [ENTSO-E Transparency Platform](https://transparency.entsoe.eu).

| Series | ENTSO-E data item | Areas used | Native resolution |
|---|---|---|---|
| Actual generation per production type | 16.1.B&C | SE (country) | 60 min (15 min where published) |
| Actual total load | 6.1.A | SE, SE1–SE4 | 60 min (15 min where published) |
| Day-ahead prices | 12.1.D | SE1–SE4; DE-AT-LU until 2018-09-30, DE-LU after | 60 min; 15 min from 2025-10-01 |
| Aggregated filling rate of water reservoirs and hydro storage plants | 16.1.D | SE, SE1–SE4 | weekly |

Two interchangeable access paths write the same normalised cache (`data.source` in
`config/base.yaml`):

* **`entsoe_api`** (default): the REST API through
  [entsoe-py](https://github.com/EnergieID/entsoe-py) 0.8.1. It needs a personal API token in
  `.env` (`ENTSOE_API_KEY=...`, see `.env.example`); check it with `uv run watervalues check-api`.
* **`filelib`**: the platform's File Library (monthly bulk exports, yearly for reservoirs),
  accessed with the website login (`uv run watervalues login` stores it in the OS credential
  store). Downloads are checksummed in `raw/filelib/_manifest.json`.

Building the weekly dataset from both sources and comparing them is a check of the pipeline itself.

**Validation only (not used by the model):** Energiföretagen's yearly weekly statistics
("Veckostatistik", 2019–2025, from its
[statistics archive](https://www.energiforetagen.se/statistik/elstatistik/arkiv-aldre-statistik/)),
with the end-of-week reservoir filling, hydro production and inflow of each ISO week.
`watervalues validate` downloads the workbooks into `external/energiforetagen/` and writes the
comparison to `processed/validation_energiforetagen.txt`, together with the workbooks' SHA-256
digests.

### Retrieval

```bash
uv run watervalues fetch              # all datasets, from data.source
uv run watervalues build --strict     # weekly_se.csv + build_report.txt
uv run watervalues validate           # validation_energiforetagen.txt
```

Without any credentials, the committed `processed/weekly_se.csv` is enough to run the model and
the notebooks.

## Licence and attribution

Source of all series: **ENTSO-E Transparency Platform**. The platform's terms of use require
citing it as the source and not implying ENTSO-E's endorsement. Day-ahead prices originate from
the power exchanges; the ENTSO-E list of freely re-usable data must be checked before this
repository is made public.

The Energiföretagen workbooks (source: Energiföretagen Sverige, *Veckostatistik*) are not
redistributed: only statistics derived from them are committed.

## Processing rules (`weekly_se.csv`)

**Time.** Everything is in UTC. A week runs from Monday 00:00 UTC to the next Monday 00:00 UTC
(ISO 8601 weeks) for ISO years 2015–2025. The model has 52 stages per year, so in ISO years with
53 weeks (2015, 2020) week 53 is folded into week 52, whose `hours` becomes 336. Downstream code
rescales energies of folded weeks to 168 hours with `per_stage()`.

**Interval data.** Energy is integrated by time: each value covers the interval given by its
resolution (60 or 15 minutes), never a count of rows. When both resolutions exist for an hour,
the one covering more of the hour is used (the finer one on ties). Revised values keep the latest
publication.

**Coverage.** A week with at least 95 % of its hours covered is scaled by
`hours / covered hours` (gaps filled with the week's mean); below that it is missing (NaN).

**Generation categories** (ENTSO-E production-type codes):
hydro = B10 + B11 + B12 (gross output; pumping consumption is reported, not subtracted),
nuclear = B14, wind = B18 + B19, solar = B16, other = every remaining type. For Sweden, ENTSO-E
reports all hydro as water reservoir (B12); fossil gas (B04), marine (B13) and solar (B16) start
on 2021-12-14 23:00 UTC, and marine stops on 2022-02-13 (the build report lists the first and
last interval of every type). These types are declared in `data.partial_period_types`: a week
outside their reporting span counts as zero when other types were reported. Any other type
that is absent for a week makes its category missing, so a major type that is published late
can never turn into a silent zero.

**Load.** `load_gwh` is the Swedish country load; SE1–SE4 loads serve as price weights and as a
cross-check (their sum is compared with the country load in the build report).

**Prices.** Zone and German prices are time-weighted weekly means. `price_se_eur_mwh` weights
the four zone prices by their hourly loads, using only hours in which every zone has a price and
a load. The German series is DE-AT-LU before 2018-09-30 22:00 UTC and DE-LU after.

**Reservoir.** See the next section. A single missing weekly value is interpolated linearly
between its neighbours and listed in the build report (2016-W39 is missing for SE and SE1, and
2020-W53 for every area). The interpolation affects one week boundary; the inflow over the two
weeks around it is unchanged.

## Reservoir timestamp convention (verified)

ENTSO-E publishes one value per ISO week without defining whether it is the storage at the start
of the week, at its end, or a weekly mean (16.1.D: "no standard method"). A one-week
misalignment would corrupt every inflow `I = storage_end - storage + hydro`. Three checks were
made:

1. **External reference.** Energiföretagen, *Kraftläget i Sverige*, week 43/2026
   (19–25 October), page 1, reports the energy content at the end of the week ("vid veckans
   slut"): SE 25 339 GWh, SE1 10 447, SE2 12 976, SE3 1 872, SE4 44 GWh. The ENTSO-E values
   published for 2026-W43 are identical in all five areas, and the ratio of the values published
   for 2026-W42 and W43 agrees, within rounding, with the report's filling at the start and at
   the end of the week (74.7 % and 75.5 %). The build re-checks these references on every run.
2. **Energiföretagen weekly statistics, 2019–2025** (`processed/validation_energiforetagen.txt`).
   - The filling at the end of workbook week *w* matches the ENTSO-E value labelled *w*: the
     median absolute difference is 95.8 GWh, against 670.3 GWh for the label *w − 1* and
     688.6 GWh for *w + 1*. The remaining difference is mostly one of scale, not timing: dividing
     the ENTSO-E values by the workbook filling gives a capacity of 33 627–33 684 GWh in
     2019–2021 and 33 945–34 026 GWh in 2022–2025 (yearly medians), not the 33 748 GWh printed
     in the workbooks.
   - Weekly hydro production agrees: correlation 0.993 in the same week (0.634 and 0.647 one week
     off) and an annual ratio ENTSO-E / Energiföretagen between 1.001 and 1.039. The UTC week bins
     therefore match the workbooks' weeks.
   - The published weekly inflow is reproduced by the end-of-week convention (correlation 0.989,
     RMSE 163 GWh/week; in weeks 2–14, 0.941 and 76 GWh/week), far better than by a weekly mean
     (0.931 and 375; 0.618 and 192) or by start-of-week values (0.786 and 663; 0.205 and 370).
     End of week has the lowest error in every year, in both windows.
   - The workbooks leave the filling and inflow of 2020-W51 to W53 and 2021-W52 empty (zero) and
     print fillings above 100 % for 2022-W38 and W39 (apparently a lost decimal point); those
     weeks are skipped and listed in the report.
3. **Physics of winter weeks** (`processed/build_report.txt`). In ISO weeks 2–14 inflow is small
   and smooth while production follows demand, so the right convention cancels production in the
   implied inflow. The week-to-week roughness of the implied winter inflow is 142.0 GWh for end of
   week against 257.8 GWh for the same labels shifted by one week (start of week), a ratio of
   0.55; the build fails above 0.7. Roughness cannot rule out a weekly mean (131.7 GWh): the
   end-of-week inflow differences two consecutive labels, so white noise in the reported storage
   enters the variance of its weekly changes six times, while the weekly-mean estimator differences
   labels two weeks apart and carries it once. With calm production, a synthetic end-of-week
   reservoir with 60 GWh of reporting noise makes roughness prefer the weekly mean
   (`tests/test_reservoir.py`). Checks 1 and 2 decide that case.

Conclusion: **the value published for week *w* is the storage at the end of week *w***. Hence
`storage_gwh(w)` = value published for *w − 1* and `storage_end_gwh(w)` = value published for *w*.
Weekly values refer to Monday 00:00 Swedish time; they are placed at Monday 00:00 UTC, a one- or
two-hour offset that is negligible at weekly resolution.

## Column dictionary

| Column | Unit | Description |
|---|---|---|
| `week_start` | date (UTC) | Monday of the ISO week |
| `iso_year`, `iso_week` | – | ISO week identifiers (week 53 folded into 52) |
| `hours` | h | Length of the stage: 168, or 336 for a folded week 52 |
| `storage_gwh` | GWh | Reservoir content at the start of the week |
| `storage_end_gwh` | GWh | Reservoir content at the end of the week (= next week's start) |
| `hydro_gwh` | GWh | Hydro production (B10 + B11 + B12) |
| `load_gwh` | GWh | Actual total load, Sweden |
| `nuclear_gwh`, `wind_gwh`, `solar_gwh`, `other_gwh` | GWh | Non-hydro generation |
| `price_se_eur_mwh` | EUR/MWh | Day-ahead price, load-weighted over SE1–SE4 |
| `price_se1_eur_mwh` … `price_se4_eur_mwh` | EUR/MWh | Day-ahead price per bidding zone |
| `price_de_eur_mwh` | EUR/MWh | Day-ahead price, DE-AT-LU then DE-LU |

## Known gaps

Missing values in `weekly_se.csv` are listed with their reason in `known_gaps.yaml`;
`watervalues build --strict` fails on any missing value that is not documented there. Nothing is
imputed in the weekly dataset; how the model treats these weeks is decided with the inflow
reconstruction (M2).

| Weeks | Missing columns | Reason |
|---|---|---|
| 2015-W01 | all generation, load, DE price | the week starts on 2014-12-29, before the Transparency Platform data |
| 2015-W50 | hydro, load, nuclear, other, SE price | ENTSO-E lacks SE load and generation for part of the week |
| 2021-W50 | solar, other | solar, fossil gas and marine are first reported during the week |
| 2022-W06 | other | marine (always zero) is last reported during the week, with gaps |

## Limitations

* Spill is not observed, so reconstructed inflows are underestimated in very wet weeks.
* The reservoir series covers regulated storage, while hydro production includes run-of-river.
* Solar is a structural zero before 2021-W50 because ENTSO-E did not report it, not because
  Sweden had none.
* Generation per type need not cover every small or industrial plant; residual demand and net
  import inherit any such gap.
* Net import `N = residual demand - hydro` ignores pumping, losses and the gap between the
  country load and the sum of the zones.
