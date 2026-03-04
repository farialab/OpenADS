"""Domain models for visualization specifications.

Pure data classes defining how visualizations should be generated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class SliceSelection:
    """Selected slices for visualization.

    Represents which axial slices to include in a visualization,
    along with the strategy used to select them.

    Attributes:
        indices: List of slice indices (0-based)
        selection_strategy: Description of how slices were selected
            (e.g., "stroke_based", "regular_interval", "combined")
    """

    indices: List[int]
    selection_strategy: str

    def __post_init__(self):
        """Validate slice selection."""
        if not self.indices:
            raise ValueError("Slice indices cannot be empty")

        # Ensure indices are sorted and unique
        if len(set(self.indices)) != len(self.indices):
            # Remove duplicates and sort
            self.indices = sorted(set(self.indices))

        # Validate indices are non-negative
        if any(idx < 0 for idx in self.indices):
            raise ValueError("Slice indices must be non-negative")

    @property
    def num_slices(self) -> int:
        """Number of selected slices."""
        return len(self.indices)

    @property
    def slice_range(self) -> tuple[int, int]:
        """Get min and max slice indices."""
        return (min(self.indices), max(self.indices))


@dataclass
class VisualizationSpec:
    """Specification for visualization generation.

    Defines all parameters needed to generate a visualization,
    without any I/O or rendering logic.

    Attributes:
        subject_id: Subject identifier
        output_path: Path where visualization will be saved
        slice_selection: Which slices to visualize
        modality: Imaging modality ("dwi" or "pwi")
        space: Image space ("native" or "MNI")
        include_predictions: Whether to include model predictions in visualization
        title: Optional custom title for the visualization
    """

    subject_id: str
    output_path: Path
    slice_selection: SliceSelection
    modality: str
    space: str = "MNI"
    include_predictions: bool = False
    title: str = None

    def __post_init__(self):
        """Validate visualization specification."""
        # Validate modality
        valid_modalities = ["dwi", "pwi"]
        if self.modality not in valid_modalities:
            raise ValueError(f"Modality must be one of {valid_modalities}, got {self.modality}")

        # Validate space
        valid_spaces = ["native", "MNI", "MNI152"]
        if self.space not in valid_spaces:
            raise ValueError(f"Space must be one of {valid_spaces}, got {self.space}")

        # Convert output_path to Path if needed
        if not isinstance(self.output_path, Path):
            self.output_path = Path(self.output_path)

        # Set default title if not provided
        if self.title is None:
            self.title = f"{self.subject_id} - {self.modality.upper()} Visualization ({self.space} space)"

    @property
    def filename(self) -> str:
        """Get output filename from path."""
        return self.output_path.name

    @property
    def output_dir(self) -> Path:
        """Get output directory from path."""
        return self.output_path.parent
