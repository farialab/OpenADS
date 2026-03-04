"""Domain models for PWI registration module."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class PWIRegistrationInputs:
    """PWI-specific input files for coregistration."""
    ttp: Path
    hp_manual: Optional[Path]
    pwi: Path  # 4D time series
    pwi_mask: Path
    dwi_mask: Path  # From DWI preprocessing


@dataclass(frozen=True)
class PWICoregResult:
    """Results from PWI→DWI coregistration."""
    ttp_on_dwi: Path
    hp_manual_on_dwi: Optional[Path]
    transform_fwd: Path
    transform_inv: Path
    # NOTE: pwi_on_dwi removed - only TTP and HP_manual processed
    ttp_masked: Path
    hp_manual_masked: Optional[Path]
    # NOTE: pwi_masked removed - only TTP and HP_manual processed


@dataclass(frozen=True)
class PWIMNIResult:
    """Results from DWI→MNI registration (via transform reuse or copy)."""
    ttp_aff: Path
    ttp_aff_norm: Optional[Path]
    ttp_affsyn: Path
    hp_manual_aff: Optional[Path]
    hp_manual_affsyn: Optional[Path]
    # NOTE: pwi_aff and pwi_affsyn removed - only TTP and HP_manual processed 
    from_copy: bool = False  # True if copied from DWI registration


@dataclass(frozen=True)
class PWIRegistrationResult:
    """Complete PWI registration pipeline results."""
    coreg: PWICoregResult
    mni: PWIMNIResult


@dataclass(frozen=True)
class DWITransformArtifacts:
    """DWI registration transforms to be reused."""
    aff_mat: Path
    warp: Path
    syn_mat: Path
    dwi_mask_template: Path
    adc_template: Path
