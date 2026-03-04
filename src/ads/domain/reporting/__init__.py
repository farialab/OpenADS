"""Domain models for reporting subsystem.

Pure data classes with no dependencies on external libraries (except dataclasses, typing, numpy).
These models represent the core business entities for reporting.
"""

from .qfv_data import AtlasFeatures, QFVResult
from .atlas_qfv_data import (
    VascularQFVResult,
    LobeQFVResult,
    AspectsQFVResult,
    AspectsPCQFVResult,
    VentriclesQFVResult,
    AllQFVResults,
)
from .report_data import (
    LesionMetrics,
    AAModelPrediction,
    RadiologyReport,
    InterpretationReport,
)
from .visualization_spec import SliceSelection, VisualizationSpec

__all__ = [
    # QFV data (legacy)
    "AtlasFeatures",
    "QFVResult",
    # Individual atlas QFV results
    "VascularQFVResult",
    "LobeQFVResult",
    "AspectsQFVResult",
    "AspectsPCQFVResult",
    "VentriclesQFVResult",
    "AllQFVResults",
    # Report data
    "LesionMetrics",
    "AAModelPrediction",
    "RadiologyReport",
    "InterpretationReport",
    # Visualization
    "SliceSelection",
    "VisualizationSpec",
]
