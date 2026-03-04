"""Model loading adapter for segmentation.

Wraps existing model loading logic without modification.
"""
from pathlib import Path
import torch
import torch.nn as nn
from ads.domain.model_config import ModelConfig


class ModelLoader:
    """
    Unified model loading adapter.

    Wraps existing model implementations in src/ads/models/
    WITHOUT modifying them.
    """

    @staticmethod
    def load_model(config: ModelConfig) -> nn.Module:
        """
        Load model from configuration.

        This wraps existing model loading without modifying model code.
        Supports both DWI and PWI modalities with different architectures.

        Args:
            config: ModelConfig specifying modality, architecture and weights

        Returns:
            Loaded PyTorch model in eval mode
        """
        # Import model classes based on modality and architecture
        if config.modality == 'dwi' and config.architecture == 'dagmnet':
            from ads.models.dagmnet_dwi import DAGMNet
            model = DAGMNet()

        elif config.modality == 'pwi' and config.architecture == 'dagmnet':
            from ads.models.dagmnet_pwi import ReplicatedDAGMNet
            model = ReplicatedDAGMNet(in_ch=config.n_channels)

        elif config.architecture == 'unet3d':
            # UNet3D works for both DWI and PWI
            from ads.models.unet3d_pwi import UNet3D

            model = UNet3D(
                in_channels=config.n_channels,
                out_channels=1,
                init_features=32
            )

        else:
            raise ValueError(
                f"Unknown model combination: modality={config.modality}, "
                f"architecture={config.architecture}"
            )

        # Load weights
        device = torch.device('cpu')  # Load to CPU first
        state_dict = torch.load(
            config.weights_path,
            map_location=device,
            weights_only=True
        )
        model.load_state_dict(state_dict)
        model.eval()

        return model

    @staticmethod
    def load_from_path(model_path: Path, architecture: str = 'dagmnet') -> nn.Module:
        """
        Legacy wrapper for backward compatibility.

        Uses existing pt_load_model from models/wrappers.py
        """
        from ads.models.wrappers import pt_load_model
        return pt_load_model(str(model_path))
