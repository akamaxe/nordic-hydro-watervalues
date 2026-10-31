"""Definitional constants.

These are conversion factors and calendar facts fixed by definition, not model parameters. Model
parameters live in ``config/base.yaml``.
"""

from typing import Final

MWH_PER_GWH: Final = 1_000.0  # [MWh/GWh]
GWH_PER_TWH: Final = 1_000.0  # [GWh/TWh]
EUR_PER_KEUR: Final = 1_000.0  # [EUR/kEUR]; GWh x EUR/MWh = MWH_PER_GWH / EUR_PER_KEUR kEUR
MINUTES_PER_HOUR: Final = 60  # [min/h]
HOURS_PER_DAY: Final = 24  # [h/day]
DAYS_PER_WEEK: Final = 7  # [day/week]
HOURS_PER_WEEK: Final = HOURS_PER_DAY * DAYS_PER_WEEK  # [h/week] in a standard ISO week
MODEL_WEEKS_PER_YEAR: Final = 52  # stages per year; ISO week 53 is folded into week 52
