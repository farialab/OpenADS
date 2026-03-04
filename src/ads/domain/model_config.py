"""Model configuration for segmentation.

Defines model loading parameters and capabilities.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration for segmentation.

    Specifies model architecture, weights, and requirements.
    """
    name: str  # 'DAGMNet_DWI', 'DAGMNet_PWI', 'UNet3D_PWI'
    weights_path: Path
    n_channels: int  # 2 or 3
    modality: Literal['dwi', 'pwi']
    architecture: Literal['dagmnet', 'unet3d']
    template_dir: Path  # For Prob_IS computation

    @property
    def requires_prob_is(self) -> bool:
        """Check if model requires Prob_IS channel (3-channel models)."""
        return self.n_channels == 3

    def __post_init__(self):
        """Validate configuration."""
        if not isinstance(self.weights_path, Path):
            object.__setattr__(self, 'weights_path', Path(self.weights_path))

        if not isinstance(self.template_dir, Path):
            object.__setattr__(self, 'template_dir', Path(self.template_dir))

        if self.n_channels not in [2, 3]:
            raise ValueError(f"n_channels must be 2 or 3, got {self.n_channels}")

        if self.modality not in ['dwi', 'pwi']:
            raise ValueError(f"modality must be 'dwi' or 'pwi', got {self.modality}")

        if self.architecture not in ['dagmnet', 'unet3d']:
            raise ValueError(f"architecture must be 'dagmnet' or 'unet3d', got {self.architecture}")
