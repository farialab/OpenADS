"""Preprocessing specification and configuration.

Defines the interface for preprocessing parameters and options.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class PreprocessingSpec:
    """Specification for preprocessing operations.

    This class encapsulates all configuration parameters for the preprocessing
    pipeline, making it easy to pass around and validate.

    Attributes:
        b_value: B-value for DWI acquisition (default: 1000)
        force_adc_calculation: Force ADC recalculation even if file exists
        check_dwi_b0_order: Automatically detect and correct DWI/B0 order
        normalization_percentile_low: Lower percentile for normalization (default: 1)
        normalization_percentile_high: Upper percentile for normalization (default: 99)
        pad_target_shape: Target shape for padding (default: (192, 224, 192))
        epsilon: Small value to avoid division by zero (default: 1e-10)
    """

    b_value: float = 1000.0
    force_adc_calculation: bool = False
    check_dwi_b0_order: bool = True
    normalization_percentile_low: float = 1.0
    normalization_percentile_high: float = 99.0
    pad_target_shape: tuple[int, int, int] = (192, 224, 192)
    epsilon: float = 1e-10

    def __post_init__(self):
        """Validate preprocessing specification."""
        if self.b_value <= 0:
            raise ValueError("b_value must be positive")
        if not (0 < self.normalization_percentile_low < self.normalization_percentile_high < 100):
            raise ValueError("Invalid normalization percentiles")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
