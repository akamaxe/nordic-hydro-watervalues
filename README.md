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

## Data

Source: ENTSO-E Transparency Platform. See [data/README.md](data/README.md) for provenance,
processing rules and known gaps.

## License

MIT — see [LICENSE](LICENSE).
