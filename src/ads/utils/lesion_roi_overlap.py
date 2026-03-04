#!/usr/bin/env python3
"""
Lesion-ROI overlap counting utility.

Calculate and report the overlap between lesion segmentations and atlas ROIs.
"""
import argparse
import logging
import os
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from typing import List, Union, Optional
from concurrent.futures import ProcessPoolExecutor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger('lesion_roi')

def load_nifti(file_path: Union[str, Path]) -> nib.Nifti1Image:
    """Load a NIfTI file."""
    return nib.as_closest_canonical(nib.load(file_path))

class LesionROICounter:
    """Count overlap between lesion segmentations and atlas ROIs."""

    def __init__(self, atlas_path: Union[str, Path]):
        """Initialize with atlas data."""
        atlas_img = load_nifti(atlas_path)
        self.atlas_data = np.round(atlas_img.get_fdata())
        self.max_roi = int(np.max(self.atlas_data))
        
    def count_lesion(self, stroke_path: Union[str, Path]) -> List[int]:
        """Count lesion voxels in each ROI for a given stroke image."""
        if not os.path.exists(stroke_path):
            logger.warning(f"Missing: {stroke_path}")
            return []

        stroke_img = load_nifti(stroke_path)
        stroke_data = stroke_img.get_fdata()
        
        if stroke_data.shape != self.atlas_data.shape:
            logger.warning(f"Shape mismatch: {stroke_path} {stroke_data.shape} vs atlas {self.atlas_data.shape}")
            return []

        roi_counts = []
        for roi_idx in range(1, self.max_roi + 1):
            count = np.sum((np.isclose(stroke_data, 1, atol=1e-3)) & (self.atlas_data == roi_idx))
            roi_counts.append(count)

        return roi_counts
    
    def process_sequential(self, stroke_paths: List[Union[str, Path]], 
                           output_csv: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """Process multiple stroke images sequentially."""
        results = []
        filenames = []
        
        for path in stroke_paths:
            path = Path(path)
            counts = self.count_lesion(path)
            if counts:  # Only add if we got valid counts
                results.append(counts)
                filenames.append(path.name)
        
        if not results:
            logger.warning("No valid results found")
            return pd.DataFrame()
            
        columns = [f"ROI_{i}" for i in range(1, self.max_roi + 1)]
        df = pd.DataFrame(results, index=filenames, columns=columns)
        
        if output_csv:
            df.to_csv(output_csv)
            
        return df
    
    def process_parallel(self, stroke_paths: List[Union[str, Path]], 
                        output_csv: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """Process multiple stroke images in parallel."""
        paths = [Path(p) for p in stroke_paths]
        
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(self.count_lesion, paths))
        
        # Filter out empty results and get corresponding filenames
        valid_results = []
        filenames = []
        
        for path, result in zip(paths, results):
            if result:
                valid_results.append(result)
                filenames.append(path.name)
        
        if not valid_results:
            logger.warning("No valid results found")
            return pd.DataFrame()
            
        columns = [f"ROI_{i}" for i in range(1, self.max_roi + 1)]
        df = pd.DataFrame(valid_results, index=filenames, columns=columns)
        
        if output_csv:
            df.to_csv(output_csv)
            
        return df


def main():
    parser = argparse.ArgumentParser(description="Count lesion-ROI overlaps")
    parser.add_argument("--atlas", required=True, help="Path to atlas NIfTI file")
    parser.add_argument("--strokes", required=True, nargs="+", help="Paths to stroke NIfTI files")
    parser.add_argument("--output", help="Path to output CSV file")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing")
    args = parser.parse_args()
    
    counter = LesionROICounter(args.atlas)
    
    if args.parallel:
        results = counter.process_parallel(args.strokes, args.output)
    else:
        results = counter.process_sequential(args.strokes, args.output)
    
    print(f"Processed {len(results)} stroke images")
    
if __name__ == "__main__":
    main()