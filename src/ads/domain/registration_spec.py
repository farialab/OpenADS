"""Registration configuration specifications."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class AffineSpec:
    """Affine registration parameters."""
    type_of_transform: str = 'Affine'
    reg_iterations: List[int] = field(default_factory=lambda: [100, 50, 25])
    shrink_factors: List[int] = field(default_factory=lambda: [8, 4, 2])
    smoothing_sigmas: List[float] = field(default_factory=lambda: [3.0, 2.0, 1.0])
    grad_step: float = 0.1
    metric: str = 'mattes'
    sampling: int = 32
    verbose: bool = True


@dataclass
class SyNSpec:
    """SyN registration parameters."""
    type_of_transform: str = 'SyN'
    reg_iterations: List[int] = field(default_factory=lambda: [70, 50, 20])
    shrink_factors: List[int] = field(default_factory=lambda: [8, 4, 2])
    smoothing_sigmas: List[float] = field(default_factory=lambda: [3.0, 2.0, 1.0])
    grad_step: float = 0.1
    metric: str = 'mattes'
    sampling: int = 32
    verbose: bool = True


@dataclass
class RegistrationSpec:
    """Complete registration configuration."""
    affine: AffineSpec
    syn: SyNSpec
    write_manifest: bool = True
