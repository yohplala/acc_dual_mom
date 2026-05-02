"""PEA Accelerated Dual Momentum framework."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("pea-momentum")
except PackageNotFoundError:  # editable install / not yet built
    __version__ = "0.0.0+unknown"
