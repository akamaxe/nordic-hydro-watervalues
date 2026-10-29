"""Water values for the aggregated Swedish hydro reservoir (EMPS-style weekly SDP)."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("nordic-hydro-watervalues")
except PackageNotFoundError:  # pragma: no cover - only when the package is not installed
    __version__ = "0.0.0"

__all__ = ["__version__"]
