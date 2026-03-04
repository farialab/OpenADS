"""Domain models for registration module."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class TemplatePaths:
    """Template file paths for registration."""
    dwi_mask_template: Path
    adc_template: Path


@dataclass(frozen=True)
class RegistrationInputs:
    """Subject input files for registration."""
    dwi_brain: Path
    adc_brain: Path
    mask: Path
    b0_brain: Optional[Path] = None
    stroke: Optional[Path] = None


@dataclass(frozen=True)
class AffineResult:
    """Results from affine registration stage."""
    dwi: Path
    adc: Path
    mask: Path
    b0: Optional[Path]
    stroke: Optional[Path]
    transform_fwd: Path
    transform_inv: Path


@dataclass(frozen=True)
class SyNResult:
    """Results from SyN registration stage."""
    dwi: Path
    adc: Path
    mask: Path
    b0: Optional[Path]
    stroke: Optional[Path]
    affine_fwd: Path
    warp_fwd: Path
    affine_inv: Path
    warp_inv: Path


@dataclass(frozen=True)
class RegistrationResult:
    """Complete registration pipeline results."""
    affine: AffineResult
    syn: SyNResult
    manifest_yaml: Path
    manifest_json: Path
