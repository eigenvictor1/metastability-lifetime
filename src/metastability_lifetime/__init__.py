"""Vacuum lifetime of the SM + singlet scalar + heavy neutrino EFT.

Python port of the lifetime calculation of the Mathematica notebook
2024-11_Scan_v1.2.nb (function ``Everything``).
"""

from .constants import Settings
from .lifetime import Result, compute_point, compute_ynu_scan, ynu_values

__all__ = ["Settings", "Result", "compute_point", "compute_ynu_scan", "ynu_values"]
__version__ = "0.1.0"
