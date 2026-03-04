#!/usr/bin/env python3
"""
Atlas ROI statistics utility for labeled atlas NIfTI files.
Provides counting of voxels per ROI and mapping to ROI names.
"""
import os
import logging
import argparse
from sympy import re
import yaml
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import List, Optional, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger('atlas_roi_stats')

def load_nifti(file_path: str) -> nib.Nifti1Image:
    """Load a NIfTI file in RAS orientation."""
    return nib.as_closest_canonical(nib.load(file_path))

def get_roi_names(roi_type=None, roi_file=None):
    """
    Get ROI names from either a text file or configuration file.
    
    Args:
        roi_type: Type of ROI (e.g., 'LOBE', 'ARTERIAL', etc.)
        roi_file: Path to text file with custom ROI names (one per line)
        
    Returns:
        List of ROI names
    """
    # 1. Load from text file if provided
    if roi_file and os.path.exists(roi_file):
        try:
            with open(roi_file, 'r') as f:
                roi_names = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(roi_names)} ROI names from {roi_file}")
            return roi_names
        except Exception as e:
            logger.error(f"Error reading ROI names from {roi_file}: {e}")
    
    # 2. Load from configuration file
    if roi_type:
        try:
            config_path = Path(__file__).resolve().parents[3] / "configs" / "dwi_pipeline.yaml"
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                
                roi_list = config["metadata"]["ROI_Names_default"].get(roi_type, [[]])[0]
                if roi_list:
                    logger.info(f"Loaded {len(roi_list)} ROI names from config for {roi_type}")
                    return roi_list
                logger.warning(f"ROI type '{roi_type}' not found in config")
            else:
                logger.warning(f"Config file not found: {config_path}")
        except Exception as e:
            logger.error(f"Error loading ROI names from config: {e}")
            
class AtlasROIStats:
    """Compute and report statistics for labeled brain atlas regions."""
    
    def __init__(self, 
                atlas_path: str, 
                roi_type: Optional[str] = None,
                roi_file: Optional[str] = None):
        """
        Initialize with atlas and optional ROI names.
        
        Args:
            atlas_path: Path to atlas NIfTI file
            roi_type: ROI names list from config (e.g., "LOBE_ROI_NAMES", "ARTERIAL_ROI_NAMES")
            roi_file: Path to text file with custom ROI names (one per line)
        """
        self.atlas_img = load_nifti(atlas_path)
        self.data = self.atlas_img.get_fdata()
        self.max_label = int(np.max(self.data))
        
        # Get ROI names
        self.roi_names = self._resolve_roi_names(roi_type, roi_file)

    def _resolve_roi_names(self, roi_type: Optional[str], roi_file: Optional[str]) -> List[str]:
        """Resolve ROI names based on provided inputs or generate defaults."""
        # 1. Use custom names if provided
        try:
            if roi_file:
                loaded_names = get_roi_names(roi_file=roi_file)
                if loaded_names:
                    return loaded_names
            
            if roi_type:
                loaded_names = get_roi_names(roi_type=roi_type)
                if loaded_names:
                    return loaded_names
        except Exception as e:
            logger.error(f"Error loading ROI names: {e}")
        # 3. Default: generate numbered ROI names
        return [f"ROI_{i}" for i in range(1, self.max_label+1)]
    
    def get_roi_counts(self) -> List[int]:
        """Count voxels for each ROI label."""
        atlas_data = np.round(self.data)
        max_roi = int(self.max_label)

        counts = []
        for i in range(1, max_roi + 1):
            mask = (atlas_data == i)
            counts.append(np.sum(mask))
        return counts
    
    def get_roi_volumes_mm3(self) -> List[float]:
        """Calculate volumes in mm³ for each ROI."""
        voxel_volume = float(np.prod(self.atlas_img.header.get_zooms()[:3]))
        return [count * voxel_volume for count in self.get_roi_counts()]
    
    def write_results(self, counts, out_path: str) -> None:
        """Write ROI statistics to file (format based on extension)."""
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with out.open("w") as f:
            if out.suffix.lower() == '.csv':
                f.write("ROI_Name,Voxel_Count\n")
                for name, count in zip(self.roi_names, counts):
                    f.write(f"{name},{count}\n")
            else:  # Default to txt
                # No Atlas/Total ROIs header
                for i, (name, count) in enumerate(zip(self.roi_names, counts), 1):
                    # If roi_type was given, ROI_Name column is just roi_type string
                    roi_name_to_write = name if not isinstance(self.roi_names, str) else self.roi_names
                    f.write(f"{roi_name_to_write}: {count}\n")

        logger.info(f"Results saved to {out_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Atlas ROI voxel counter")
    parser.add_argument("--atlas", "-a", required=True, help="Atlas NIfTI file")
    parser.add_argument("--out", "-o", required=True, help="Output file (.txt or .csv)")
    parser.add_argument("--roi-names", "-n", help="Comma-separated ROI names")
    parser.add_argument("--roi-type", "-t", 
                       help="Predefined ROI type (LOBE_ROI_NAMES, ARTERIAL_ROI_NAMES, etc.)")
    args = parser.parse_args()

    roi_names = args.roi_names.split(',') if args.roi_names else None
    
    stats = AtlasROIStats(
        atlas_path=args.atlas,
        roi_names=roi_names,
        roi_names_type=args.roi_type
    )
    
    stats.write_results(args.out)


if __name__ == "__main__":
    main()
