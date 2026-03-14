"""
my_cholesky: Research implementation of RPCholesky variants
for scalable Gaussian process models.

Public API (initial):
- accelerated_rpcholesky / arpcholesky
"""

from .arpcholesky import accelerated_rpcholesky, arpcholesky
from .lra import PSDLowRank
from .matrix import AbstractPSDMatrix, PSDMatrix, KernelMatrix

__all__ = [
    "accelerated_rpcholesky",
    "arpcholesky",
    "PSDLowRank",
    "AbstractPSDMatrix",
    "PSDMatrix",
    "KernelMatrix",
]

