"""Segmentation configuration specifications.

Pure configuration data objects.
"""
from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class ProbabilityMapSpec:
    """Probability map (Prob_IS) configuration.

    Parameters for ischemic stroke probability computation.
    """
    fwhm: float = 2.0
    alpha_dwi: float = 1.5
    lambda_dwi: float = 4.0
    alpha_adc: float = 0.5
    lambda_adc: float = 2.0
    zth: float = 2.0

    @property
    def model_vars(self) -> List[float]:
        """Return as list for backward compatibility with get_Prob_IS."""
        return [
            self.fwhm,
            self.alpha_dwi,
            self.lambda_dwi,
            self.alpha_adc,
            self.lambda_adc,
            self.zth
        ]


@dataclass
class InferenceSpec:
    """Model inference configuration."""
    target_shape: Tuple[int, int, int] = (192, 224, 192)
    downsampling_factor: int = 2  # DS parameter for multi-scale inference
    mask_threshold: float = 0.5
    prediction_threshold: float = 0.5
    n_channel: int = 3  # 2 or 3 (with Prob_IS)
    device: str = 'cpu'


@dataclass
class PWIInferenceSpec(InferenceSpec):
    """PWI-specific inference configuration.

    Extends InferenceSpec with PWI-specific parameters.
    """
    z_stride_factor: int = 2  # Z-stride factor for PWI inference
    use_stroke_channel: bool = False  # Whether to use stroke as 4th channel
    n_channel: int = 3  # 3 (DWI, ADC, TTP) or 4 (+ Stroke)


@dataclass
class PostProcessingSpec:
    """Post-processing configuration."""
    apply_postprocessing: bool = True
    remove_small_objects: bool = True
    min_object_size: int = 5


@dataclass
class SegmentationSpec:
    """Complete segmentation pipeline configuration."""
    probability_map: ProbabilityMapSpec = field(default_factory=ProbabilityMapSpec)
    inference: InferenceSpec = field(default_factory=InferenceSpec)
    postprocessing: PostProcessingSpec = field(default_factory=PostProcessingSpec)
    compute_metrics: bool = True
    save_normalized_inputs: bool = True
