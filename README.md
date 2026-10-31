# nordic-hydro-watervalues

[![CI](https://github.com/akamaxe/nordic-hydro-watervalues/actions/workflows/ci.yml/badge.svg)](https://github.com/akamaxe/nordic-hydro-watervalues/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

*Water values for the aggregated Swedish hydro reservoir — a minimal, open EMPS-style
stochastic dynamic programming model built on ENTSO-E data.*

> **Status:** work in progress. Each milestone (M0–M6) is tracked as a GitHub issue.

## Reproduce

```bash
uv sync
uv run pytest
```

## Notebooks

1. [Data overview](notebooks/01_data_overview.ipynb): the weekly dataset and how the reservoir
   timestamp convention was verified.
2. [Inflow reconstruction](notebooks/02_inflow_reconstruction.ipynb): weekly inflows from the
   reservoir energy balance, dry and wet years, and the inflow scenarios of the model.
3. [Price curve](notebooks/03_price_model.ipynb): the reduced-form curve linking the Swedish
   price to net import and the German price, eight variants compared out of sample, and the
   weekly cost function of the water value model.
4. [Water values](notebooks/04_water_values.ipynb): the stochastic dynamic programme, the
   water value of every week and storage level, and checks against theory and the grid.
5. [Backtest and scenarios](notebooks/05_backtest_and_scenarios.ipynb): each year of
   2015–2025 simulated with water values that have not seen it, the continental price
   scenarios, and the driest against the wettest year.

## Data

Source: ENTSO-E Transparency Platform. See [data/README.md](data/README.md) for provenance,
processing rules and known gaps.

## License

MIT — see [LICENSE](LICENSE).
