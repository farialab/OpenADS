"""NIfTI I/O adapters.

Provides consistent interfaces for loading and saving NIfTI files with
proper orientation handling.
"""

from .loader import NiftiLoader
from .saver import NiftiSaver
from .transforms import OrientationFixer, SpatialTransformer

__all__ = [
    "NiftiLoader",
    "NiftiSaver",
    "OrientationFixer",
    "SpatialTransformer",
]
