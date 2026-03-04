"""PWI registration configuration specifications."""
from dataclasses import dataclass


@dataclass
class PWICoregSpec:
    """PWI→DWI coregistration parameters."""
    type_of_transform: str = 'Affine'
    verbose: bool = True


@dataclass
class PWIRegistrationSpec:
    """Complete PWI registration configuration."""
    coreg: PWICoregSpec
    output_orientation: str = "RAS"
    interpolator_images: str = "linear"
    interpolator_masks: str = "nearestNeighbor"
    skip_reregistration: bool = True  # If True, copy from DWI if available
    force_recompute: bool = False  # If True, always recompute even if DWI exists
    # If True, raw TTP is already effectively skull-stripped; use *_TTP_space-DWI.nii.gz downstream.
    # If False, use *_TTP_space-DWI_desc-brain.nii.gz downstream.
    raw_ttp_masked: bool = True
