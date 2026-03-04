"""Atlas loading adapter.

Handles file I/O for loading atlas templates and lookup tables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd


class AtlasLoader:
    """Load atlas templates and lookup tables from disk.

    Pure I/O adapter - loads files and returns numpy arrays or dataframes.

    """

    # Atlas configuration mapping
    ATLAS_CONFIG = {
        "vascular": {
            "atlas_name": "ArterialAtlas_MNI182.nii",
            "volume_table": "Vas_Lookup_Volume_df_MNI.pkl",
            "label_table": "ArterialLabelLookupTable.txt",
        },
        "lobe": {
            "atlas_name": "LobeAtlas_MNI182.nii",
            "volume_table": "Lobe_Lookup_Volume_df_MNI.pkl",
            "label_table": "LobesLabelLookupTable.txt",
        },
        "aspects": {
            "atlas_name": "AspectsAtlas_MNI182.nii",
            "volume_table": "Aspects_Lookup_Volume_df_MNI.pkl",
            "label_table": "AspectsLabelLookupTable.txt",
        },
        "aspectpc": {
            "atlas_name": "AspectsPcaAtlas_MNI182.nii",
            "volume_table": "Aspects_PCA_Lookup_Volume_df_MNI.pkl",
            "label_table": "AspectsPCALabelLookupTable.txt",
        },
        "Ventricles": {
            "atlas_name": "VentriclesEnlargedAtlas_MNI182.nii",
            "volume_table": None,
            "label_table": None,
        },
        "watershed": {
            "atlas_name": "WatershedAtlas_MNI182.nii",
            "volume_table": None,
            "label_table": None,
        },
        "bpm_type1": {
            "atlas_name": "BPMTypeIV2Atlas_MNI182.nii",
            "volume_table": "JHU_MNI_SS_BPM_TypeI_V2_Lookup_Volume_df_MNI.pkl",
            "label_table": "BPMLabelLookupTable.txt",
        },
        "bmos": {
            "atlas_name": "BMOSAtlas_MNI182.nii",
            "volume_table": "BMOS_Lookup_Volume_df_MNI.pkl",
            "label_table": None,
        },
        "bmis": {
            "atlas_name": "BMISAtlas_MNI182.nii",
            "volume_table": "BMIS_Lookup_Volume_df_MNI.pkl",
            "label_table": None,
        },
    }

    def __init__(self, atlas_dir: Path):
        """Initialize atlas loader.

        Args:
            atlas_dir: Directory containing atlas files
        """
        self.atlas_dir = Path(atlas_dir)
        if not self.atlas_dir.exists():
            raise FileNotFoundError(f"Atlas directory not found: {atlas_dir}")

    def load_template(self, atlas_name: str) -> Optional[np.ndarray]:
        """Load atlas template image.

        Args:
            atlas_name: Name of atlas (e.g., "vascular", "lobe", "aspects")

        Returns:
            3D numpy array of atlas template, or None if not found

        """
        if atlas_name not in self.ATLAS_CONFIG:
            raise ValueError(f"Unknown atlas: {atlas_name}. Available: {list(self.ATLAS_CONFIG.keys())}")

        config = self.ATLAS_CONFIG[atlas_name]
        atlas_filename = config["atlas_name"]

        if not atlas_filename:
            return None

        atlas_path = self.atlas_dir / atlas_filename

        if not atlas_path.exists():
            print(f"Warning: Atlas file not found: {atlas_path}")
            return None

        try:
            # Load and convert to canonical orientation
            img = nib.load(str(atlas_path))
            img_canonical = nib.as_closest_canonical(img)
            data = np.squeeze(img_canonical.get_fdata())
            return data
        except Exception as e:
            print(f"Error loading atlas {atlas_name}: {e}")
            return None

    def load_volume_table(self, atlas_name: str) -> Optional[pd.DataFrame]:
        """Load atlas volume lookup table.

        Args:
            atlas_name: Name of atlas

        Returns:
            DataFrame with volume lookup table, or None if not found
        """
        if atlas_name not in self.ATLAS_CONFIG:
            raise ValueError(f"Unknown atlas: {atlas_name}")

        config = self.ATLAS_CONFIG[atlas_name]
        volume_table_filename = config["volume_table"]

        if not volume_table_filename:
            return None

        table_path = self.atlas_dir / volume_table_filename

        if not table_path.exists():
            print(f"Warning: Volume table not found: {table_path}")
            return None

        try:
            df = pd.read_pickle(str(table_path))
            return df
        except Exception as e:
            print(f"Error loading volume table for {atlas_name}: {e}")
            return None

    def load_all_atlases(
        self,
        atlas_list: List[str],
    ) -> Tuple[Dict[str, Optional[np.ndarray]], Dict[str, Optional[pd.DataFrame]]]:
        """Load all required atlases and their volume tables.

        Args:
            atlas_list: List of atlas names to load

        Returns:
            Tuple of (template_dict, volume_table_dict)
        """
        templates = {}
        volume_tables = {}

        for atlas_name in atlas_list:
            templates[atlas_name] = self.load_template(atlas_name)
            volume_tables[atlas_name] = self.load_volume_table(atlas_name)

        return templates, volume_tables

    def get_available_atlases(self) -> List[str]:
        """Get list of available atlas names.

        Returns:
            List of atlas names that have files in the atlas directory
        """
        available = []
        for atlas_name, config in self.ATLAS_CONFIG.items():
            atlas_filename = config["atlas_name"]
            if atlas_filename:
                atlas_path = self.atlas_dir / atlas_filename
                if atlas_path.exists():
                    available.append(atlas_name)
        return available
