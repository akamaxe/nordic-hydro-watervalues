"""Definitional unit constants.

These are conversion factors fixed by definition, not model parameters. Model parameters live in
``config/base.yaml``.
"""

from typing import Final

MWH_PER_GWH: Final = 1_000.0  # [MWh/GWh]
HOURS_PER_DAY: Final = 24  # [h/day]
DAYS_PER_WEEK: Final = 7  # [day/week]
HOURS_PER_WEEK: Final = HOURS_PER_DAY * DAYS_PER_WEEK  # [h/week] in a standard ISO week
