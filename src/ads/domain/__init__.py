"""Domain layer for OpenADS - stable interfaces and data structures.

This layer defines core domain objects and specifications with no external dependencies.
It serves as the contract between different layers of the application.
"""

from .data_objects import LoadedData, PreprocessResult
from .preprocessing_spec import PreprocessingSpec

__all__ = [
    "LoadedData",
    "PreprocessResult",
    "PreprocessingSpec",
]
