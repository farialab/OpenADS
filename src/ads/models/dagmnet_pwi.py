#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DAGMNet inference module for PWI hypoperfusion segmentation.
Provides single-subject prediction interface for pipeline integration.

Usage:
    from ads.models.dagmnet_pwi import DAGMNetPredictor
    
    predictor = DAGMNetPredictor(
        checkpoint_path="/path/to/best_model.pth",
        n_channels=4,
        device="cuda:0"
    )
    
    pred_mask = predictor.predict_subject(
        subject_dir="/path/to/subject",
        subject_id="sub-001"
    )
"""

import os
import warnings
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from contextlib import nullcontext
import scipy
from scipy import ndimage as scin
from skimage.segmentation import slic
from skimage import filters


# ============================================================================
# Configuration Constants
# ============================================================================
ORIG_SHAPE = (182, 218, 182)
PAD_SHAPE = (192, 224, 192)
DS = 2
Z_STRIDE_FACTOR = 2
POST_MIN_SIZE = 50
DEFAULT_THRESH = 0.49


# ============================================================================
# HP_symClassic Generator
# ============================================================================
class HypoperfusionGenerator:
    """
    Generates hypoperfusion map using symmetry-based classical method.
    Used as fallback when HP_symClassic.nii.gz is not found.
    """
    
    def __init__(self, subj_dir: str, subject_id: str = None, **kwargs):
        if not os.path.isdir(subj_dir):
            raise ValueError(f"Subject directory not found: {subj_dir}")
        self.subj_dir = subj_dir
        self.subj_id = subject_id or os.path.basename(os.path.normpath(subj_dir))
        
        # Default parameters
        self.params = {
            'ADC_th': 2.3,
            'sigma_th': 2,
            'sym_size': 7,
            'Sym_th': 0.7,
            'remove_max_size': 10,
            'n_segments': 300,
            'compactness': 20,
            'sigma': 3,
            'start_label': 1,
            'percent_th': 0.2
        }
        self.params.update(kwargs)
        
        # Define file paths
        self._define_paths()
    
    def _define_paths(self):
        pwi_root = self.subj_dir
        if os.path.basename(pwi_root) == "registration":
            pwi_root = os.path.dirname(pwi_root)
        seg_dir = os.path.join(pwi_root, "segmentation")
        os.makedirs(seg_dir, exist_ok=True)

        paths = self._find_files_path(self.subj_dir, self.subj_id)
        self.paths = {
            "ADC_path": paths.get("adc_mni"),
            "ADCMask_path": paths.get("adc_mask"),
            "TTP_path": paths.get("ttp_mni"),
            "HP_symClassic_path": None,
        }

        self.paths["HP_symClassic_path"] = os.path.join(
            seg_dir, f"{self.subj_id}_HPsymclassic-mask_space-MNI152.nii.gz"
        )

    @staticmethod
    def _build_search_dirs(subject_dir: str) -> list[str]:
        """Build prioritized search directories: PWI/registration -> DWI/registration -> local fallbacks."""
        p = Path(subject_dir).expanduser().resolve()
        if p.name == "registration":
            pwi_root = p.parent
        elif p.name == "PWI":
            pwi_root = p
        else:
            pwi_root = p

        subject_root = pwi_root.parent if pwi_root.name in {"PWI", "DWI"} else pwi_root
        candidates = [
            pwi_root / "registration",
            subject_root / "DWI" / "registration",
            p,
            pwi_root,
            pwi_root / "segmentation",
            pwi_root / "preprocess",
        ]

        seen = set()
        ordered = []
        for c in candidates:
            key = str(c)
            if key not in seen:
                ordered.append(key)
                seen.add(key)
        return ordered
    
    @staticmethod
    def _find_files_path(subject_dir: str, subject_id: str) -> Dict[str, str]:
        """Find input files with flexible naming matching actual registration output."""
        subject_id_clean = subject_id.replace("sub-", "")
        
        # Search priority: PWI/registration -> DWI/registration -> local fallbacks
        search_dirs = HypoperfusionGenerator._build_search_dirs(subject_dir)
        
        # Mapping key to potential filename fragments
        # Note: Your files use _space-MNI152_aff.nii.gz or _aff_desc-norm.nii.gz
        possible_files = {
            'adc_mni': [
                f"{subject_id}_ADC_space-MNI152_aff_desc-norm.nii.gz",
                f"{subject_id}_ADC_space-MNI152_aff.nii.gz",
                f"sub-{subject_id_clean}_ADC_space-MNI152_aff_desc-norm.nii.gz",
                f"sub-{subject_id_clean}_ADC_space-MNI152_aff.nii.gz",
            ],
            'adc_mask': [
                f"{subject_id}_DWIbrain-mask_space-MNI152_aff.nii.gz",
                f"sub-{subject_id_clean}_DWIbrain-mask_space-MNI152_aff.nii.gz",
            ],
            'ttp_mni': [
                f"{subject_id}_TTP_space-MNI152_aff_desc-norm.nii.gz",
                f"{subject_id}_TTP_space-MNI152_aff.nii.gz",
                f"sub-{subject_id_clean}_TTP_space-MNI152_aff_desc-norm.nii.gz",
                f"sub-{subject_id_clean}_TTP_space-MNI152_aff.nii.gz",
            ]
        }
        
        paths = {}
        for key, filenames in possible_files.items():
            for search_dir in search_dirs:
                if key in paths: break
                if not os.path.exists(search_dir): continue
                for filename in filenames:
                    fp = os.path.join(search_dir, filename)
                    if os.path.exists(fp):
                        paths[key] = fp
                        break
        return paths
        
    def _debug_print_available_files(self):
        """Print all available files in subject directory for debugging."""
        print(f"\n🔍 Debug: Files in {self.subj_dir}:")
        try:
            files = sorted(os.listdir(self.subj_dir))
            nifti_files = [f for f in files if f.endswith('.nii.gz')]
            if nifti_files:
                for f in nifti_files:
                    print(f"  - {f}")
            else:
                print("  (No .nii.gz files found)")
        except Exception as e:
            print(f"  ❌ Cannot list directory: {e}")
        print()
    
    def _load_nifti_as_ras(self, imgpath: str):
        if not imgpath or not os.path.exists(imgpath):
            return None, None
        img = nib.as_closest_canonical(nib.load(imgpath))
        data = np.squeeze(img.get_fdata()).astype(np.float64)
        return img, data
            
    def _load_img_affmat(self, img_fname_path: str):
        """Load image with affine matrix."""
        if img_fname_path is None:
            return None, None, None
        img_obj, img_data = self._load_nifti_as_ras(img_fname_path)
        return img_obj, img_data, img_obj.affine if img_obj else None
    
    def _get_new_nib_imgj(self, new_img_data: np.ndarray, template_img_obj, data_type=np.float32):
        """Create new NIfTI image from data and template."""
        template_img_obj.set_data_dtype(data_type)
        if new_img_data.dtype != np.dtype(data_type):
            new_img_data = new_img_data.astype(np.dtype(data_type))
        
        header = template_img_obj.header
        header['glmax'] = np.max(new_img_data)
        header['glmin'] = np.min(new_img_data)
        
        new_img_obj = nib.Nifti1Image(new_img_data, template_img_obj.affine, header)
        new_img_obj.header.set_slope_inter(1, 0)
        return new_img_obj
    
    def _img3d_erosion(self, mask_img: np.ndarray, structure=np.ones((3, 3))) -> np.ndarray:
        """3D binary erosion slice by slice."""
        new_mask_img = np.zeros_like(mask_img)
        for i in range(mask_img.shape[2]):
            new_mask_img[:, :, i] = scin.binary_erosion(mask_img[:, :, i], structure=structure)
        return new_mask_img
    
    def _get_ttp_mask(self, ttp_img: np.ndarray) -> np.ndarray:
        """Generate brain mask from TTP using Otsu thresholding."""
        thresh = filters.threshold_otsu(ttp_img[ttp_img > 0])
        ttp_mask = (ttp_img > thresh).astype(np.float32)
        return ttp_mask
    
    def _clear_ttp_background(self, ttp_img: np.ndarray, ttp_mask_img: np.ndarray, sigma_th: float = 2) -> np.ndarray:
        """Clear background based on standard deviation."""
        ttp_img_pro = ttp_img.copy()
        for i in range(ttp_mask_img.shape[2]):
            slice_pro = ttp_img_pro[:, :, i]
            slice_orig = ttp_img[:, :, i]
            slice_mask = ttp_mask_img[:, :, i]
            
            masked_data = slice_orig[slice_mask > 0.5]
            if masked_data.size > 0:
                threshold = np.mean(masked_data) - sigma_th * np.std(masked_data)
                slice_pro[slice_orig < threshold] = 0
            
            ttp_img_pro[:, :, i] = slice_pro
        return ttp_img_pro
    
    def _max_filter_img_2d_slice(self, img: np.ndarray, size: int = 3) -> np.ndarray:
        """Apply 2D maximum filter to each slice."""
        new_img = np.zeros_like(img)
        for i in range(img.shape[2]):
            new_img[:, :, i] = scin.maximum_filter(img[:, :, i], size)
        return new_img
    
    def _remove_small_objects(self, img: np.ndarray, remove_max_size: int = 5, structure=np.ones((3, 3))) -> np.ndarray:
        """Remove small connected components."""
        binary = (img > 0).astype(int)
        labels, num_features = scipy.ndimage.label(binary, structure=structure)
        
        new_img = img.copy()
        component_sizes = np.bincount(labels.ravel())
        
        small_components = [i for i, size in enumerate(component_sizes) if 0 < size < remove_max_size]
        
        for comp_label in small_components:
            new_img[labels == comp_label] = 0
        
        return new_img
    
    def _remove_small_objects_in_slice(self, img: np.ndarray, remove_max_size: int = 5, structure=np.ones((3, 3))) -> np.ndarray:
        """Apply small object removal to each slice."""
        new_img = np.zeros_like(img)
        for i in range(img.shape[-1]):
            new_img[:, :, i] = self._remove_small_objects(img[:, :, i], remove_max_size=remove_max_size, structure=structure)
        return new_img
    
    def _suppress_side(self, img: np.ndarray) -> np.ndarray:
        """Suppress the side with lower signal intensity."""
        m, n, s = img.shape
        mid_point = m // 2
        
        right_mask = np.zeros_like(img)
        right_mask[:mid_point, :, :] = 1
        
        left_mask = right_mask[::-1, :, :]
        
        if np.sum(img[right_mask > 0.5]) > np.sum(img[left_mask > 0.5]):
            return img * right_mask
        else:
            return img * left_mask
    
    def _sv_select_by_percent(self, segments: np.ndarray, mask_img: np.ndarray, percent_th: float = 0.2) -> np.ndarray:
        """Select supervoxels based on overlap percentage."""
        new_mask = np.zeros_like(segments, dtype=np.float32)
        for i in range(segments.shape[-1]):
            segments_slice = segments[:, :, i]
            mask_slice = mask_img[:, :, i]
            unique_segments = np.unique(segments_slice)
            
            for seg_val in unique_segments:
                if seg_val == 0:
                    continue
                segment_mask = (segments_slice == seg_val)
                
                overlap = np.sum(segment_mask[mask_slice > 0.5])
                total_size = np.sum(segment_mask)
                
                if total_size > 0 and (overlap / total_size) > percent_th:
                    new_mask[:, :, i][segment_mask] = 1
        return new_mask
    
    def _sv_segment_slic(self, ttp_img: np.ndarray, ttp_mask_img: np.ndarray, 
                         adc_mask_img: np.ndarray, adc_img: np.ndarray) -> np.ndarray:
        """Core SLIC-based segmentation logic."""
        p = self.params
        
        # Step 1: Pre-processing
        adc_mask_eroded = self._img3d_erosion(adc_mask_img, structure=np.ones((5, 5)))
        adc_threshold_mask = ((adc_img > adc_img[0, 0, 0]) & (adc_img < p['ADC_th'])) * adc_mask_eroded
        
        ttp_img_pro = ttp_img * adc_threshold_mask * ttp_mask_img
        ttp_img_pro[ttp_mask_img < 0.5] = 0
        ttp_img_pro = self._clear_ttp_background(ttp_img_pro, ttp_mask_img, sigma_th=p['sigma_th'])
        
        # Step 2: Asymmetry analysis
        ttp_img_pro_flip = ttp_img_pro[::-1, :, :]
        ttp_img_pro_flip_max = self._max_filter_img_2d_slice(ttp_img_pro_flip, size=p['sym_size'])
        
        img_diff = ttp_img_pro - ttp_img_pro_flip_max
        diff_mask = (img_diff > p['Sym_th'])
        
        # Step 3: Post-processing
        new_img = self._remove_small_objects_in_slice(diff_mask, remove_max_size=p['remove_max_size'])
        new_img = self._suppress_side(new_img)
        new_img = self._remove_small_objects_in_slice(new_img, remove_max_size=p['remove_max_size'])
        
        # Step 4: SLIC segmentation
        segments_slic = np.zeros_like(ttp_img)
        for i in range(ttp_img.shape[-1]):
            slice_3d = np.stack((ttp_img[:, :, i],) * 3, axis=-1)
            segments_slic[:, :, i] = slic(
                slice_3d,
                n_segments=p['n_segments'],
                compactness=p['compactness'],
                sigma=p['sigma'],
                start_label=p['start_label']
            )
        
        new_mask_slic = self._sv_select_by_percent(segments_slic, new_img, percent_th=p['percent_th'])
        new_mask_slic = new_mask_slic * (adc_img < p['ADC_th'])
        
        return new_mask_slic
    
    def generate(self, save_hp: bool = True) -> Optional[np.ndarray]:
        """
        Generate HP_symClassic map.
        
        Args:
            save_hp: Whether to save the result as NIfTI file
        
        Returns:
            Generated hypoperfusion mask, or None if generation fails
        """
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        # Debug: Print available files
        self._debug_print_available_files()
        
        # Check if required files exist
        required_keys = ['ADC_path', 'ADCMask_path', 'TTP_path']
        missing = [k for k in required_keys if self.paths.get(k) is None]
        
        if missing:
            print(f"❌ Missing required files for HP_symClassic generation:")
            for key in missing:
                print(f"   - {key}")
            print(f"\n💡 File detection results:")
            paths_dict = self._find_files_path(self.subj_dir, self.subj_id)
            print(f"   Search order:")
            for d in self._build_search_dirs(self.subj_dir):
                print(f"      - {d}")
            for key in ['adc_mni', 'adc_mask', 'ttp_mni']:
                status = '✓ Found' if key in paths_dict else '✗ Not found'
                file_path = paths_dict.get(key, 'N/A')
                print(f"   {key}: {status}")
                if key in paths_dict:
                    print(f"      → {os.path.basename(file_path)}")
            return None
        
        # Load required images
        try:
            print(f"Loading ADC from: {os.path.basename(self.paths['ADC_path'])}")
            adc_imgj, adc_img, _ = self._load_img_affmat(self.paths['ADC_path'])
            
            print(f"Loading ADC mask from: {os.path.basename(self.paths['ADCMask_path'])}")
            adc_mask_imgj, adc_mask_img, _ = self._load_img_affmat(self.paths['ADCMask_path'])
            
            print(f"Loading TTP from: {os.path.basename(self.paths['TTP_path'])}")
            ttp_imgj, ttp_img, _ = self._load_img_affmat(self.paths['TTP_path'])
            
            if adc_img is None or adc_mask_img is None or ttp_img is None:
                print(f"❌ Failed to load required images for {self.subj_id}")
                return None
            
            ttp_mask = self._get_ttp_mask(ttp_img)
        except Exception as e:
            print(f"❌ Error loading images for {self.subj_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # Generate HP map
        try:
            print("Generating HP_symClassic using SLIC segmentation...")
            predicted_hp_map = self._sv_segment_slic(ttp_img, ttp_mask, adc_mask_img, adc_img)
        except Exception as e:
            print(f"❌ Error generating HP_symClassic for {self.subj_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # Save if requested
        if save_hp:
            try:
                output_filename = self.paths['HP_symClassic_path']
                new_mask_imgj = self._get_new_nib_imgj(predicted_hp_map, ttp_imgj)
                nib.save(new_mask_imgj, output_filename)
                print(f"✓ Generated and saved HP_symClassic to: {output_filename}")
            except Exception as e:
                print(f"❌ Error saving HP_symClassic for {self.subj_id}: {e}")
                return None
        
        return predicted_hp_map


# ============================================================================
# Model Architecture (ReplicatedDAGMNet)
# ============================================================================
class ConvBNAct(nn.Module):
    """3D Convolution + BatchNorm + SELU activation."""
    
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm3d(out_c, eps=1e-3, momentum=0.99)
        self.act = nn.SELU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class sAG(nn.Module):
    """Spatial Attention Gate."""
    
    def __init__(self, main_c: int, aux_c: int):
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Conv3d(main_c, 1, 1),
            nn.SELU(inplace=True)
        )
        self.aux_path = nn.Sequential(
            nn.Conv3d(aux_c, 1, 1),
            nn.SELU(inplace=True)
        )
        self.conv5_main = nn.Conv3d(3, 1, 5, padding=2)
        self.conv5_aux = nn.Conv3d(3, 1, 5, padding=2)
        self.final_conv = nn.Conv3d(1, 1, 1)
        self.act = nn.SELU(inplace=True)
    
    def forward(self, main_x: torch.Tensor, aux_x: torch.Tensor) -> torch.Tensor:
        main_cat = torch.cat([
            torch.mean(main_x, 1, True),
            torch.max(main_x, 1, True)[0],
            self.main_path(main_x)
        ], 1)
        aux_cat = torch.cat([
            torch.mean(aux_x, 1, True),
            torch.max(aux_x, 1, True)[0],
            self.aux_path(aux_x)
        ], 1)
        combined = self.act(self.conv5_main(main_cat)) + self.act(self.conv5_aux(aux_cat))
        return self.final_conv(combined)


class cAG(nn.Module):
    """Channel Attention Gate."""
    
    def __init__(self, main_c: int, aux_c: int, hidden_c: int):
        super().__init__()
        self.main_avg_fc = nn.Linear(main_c, hidden_c)
        self.main_max_fc = nn.Linear(main_c, hidden_c)
        self.aux_avg_fc = nn.Linear(aux_c, hidden_c)
        self.aux_max_fc = nn.Linear(aux_c, hidden_c)
        self.fc1 = nn.Linear(hidden_c, main_c)
        self.fc2 = nn.Linear(main_c, main_c)
        self.act = nn.SELU(inplace=True)
    
    def forward(self, main_x: torch.Tensor, aux_x: torch.Tensor) -> torch.Tensor:
        main_avg = torch.mean(main_x, (2, 3, 4))
        main_max = torch.amax(main_x, (2, 3, 4))
        aux_avg = torch.mean(aux_x, (2, 3, 4))
        aux_max = torch.amax(aux_x, (2, 3, 4))
        
        combined = (
            self.act(self.main_avg_fc(main_avg)) +
            self.act(self.main_max_fc(main_max)) +
            self.act(self.aux_avg_fc(aux_avg)) +
            self.act(self.aux_max_fc(aux_max))
        )
        return self.fc2(self.act(self.fc1(combined))).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


class ReplicatedDAGMNet(nn.Module):
    """Deep Attention-Guided Multi-scale Network for 3D segmentation."""
    
    def __init__(self, in_ch: int = 4):
        super().__init__()
        
        # Multi-scale encoders
        self.enc1_block = nn.Sequential(ConvBNAct(in_ch, 32), ConvBNAct(32, 32))
        self.enc2_block = nn.Sequential(ConvBNAct(in_ch, 32), ConvBNAct(32, 32))
        self.enc3_block = nn.Sequential(ConvBNAct(in_ch, 32), ConvBNAct(32, 32))
        self.enc4_block = nn.Sequential(ConvBNAct(in_ch, 32), ConvBNAct(32, 32))
        
        self.max_pool = nn.MaxPool3d(2)
        
        # Hierarchical encoders
        self.enc12_block = nn.Sequential(ConvBNAct(64, 64), ConvBNAct(64, 64))
        self.enc23_block = nn.Sequential(ConvBNAct(96, 128), ConvBNAct(128, 128))
        self.enc34_block = nn.Sequential(ConvBNAct(160, 256), ConvBNAct(256, 256))
        
        # Attention gates
        self.dag1, self.cag1 = sAG(32, 32), cAG(32, 32, 2)
        self.dag2, self.cag2 = sAG(64, 32), cAG(64, 32, 4)
        self.dag3, self.cag3 = sAG(128, 32), cAG(128, 32, 8)
        self.dag4, self.cag4 = sAG(256, 32), cAG(256, 32, 16)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(ConvBNAct(256, 256), ConvBNAct(256, 256))
        
        # Decoder
        self.up4 = nn.ConvTranspose3d(256, 128, 2, 2)
        self.dec3 = nn.Sequential(ConvBNAct(256, 128), ConvBNAct(128, 128))
        self.up3 = nn.ConvTranspose3d(128, 64, 2, 2)
        self.dec2 = nn.Sequential(ConvBNAct(128, 64), ConvBNAct(64, 64))
        self.up2 = nn.ConvTranspose3d(64, 32, 2, 2)
        self.dec1 = nn.Sequential(ConvBNAct(64, 32), ConvBNAct(32, 32))
        
        # Deep supervision outputs
        self.out4 = nn.Conv3d(256, 1, 3, 1, 1)
        self.out3 = nn.Conv3d(128, 1, 3, 1, 1)
        self.out2 = nn.Conv3d(64, 1, 3, 1, 1)
        self.out1 = nn.Conv3d(32, 1, 3, 1, 1)
        
        # Fusion layers
        self.fuse_up_t15 = nn.ConvTranspose3d(64, 32, 2, 2)
        self.fuse_up_t12_a = nn.ConvTranspose3d(128, 64, 2, 2)
        self.fuse_up_t12_b = nn.ConvTranspose3d(64, 32, 2, 2)
        self.fuse_conv = nn.Sequential(ConvBNAct(128, 32), nn.Conv3d(32, 1, 1))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Multi-scale input encoding
        e1 = self.enc1_block(x)
        e2 = self.enc2_block(F.interpolate(x, scale_factor=0.5, mode='trilinear', align_corners=False))
        e3 = self.enc3_block(F.interpolate(x, scale_factor=0.25, mode='trilinear', align_corners=False))
        e4 = self.enc4_block(F.interpolate(x, scale_factor=0.125, mode='trilinear', align_corners=False))
        
        # Hierarchical encoding
        enc12 = self.enc12_block(torch.cat([self.max_pool(e1), e2], 1))
        enc23 = self.enc23_block(torch.cat([self.max_pool(enc12), e3], 1))
        enc34 = self.enc34_block(torch.cat([self.max_pool(enc23), e4], 1))
        
        # Attention-guided features
        a1 = e1 * torch.sigmoid(self.dag1(e1, e1)) * torch.sigmoid(self.cag1(e1, e1))
        a2 = enc12 * torch.sigmoid(self.dag2(enc12, e2)) * torch.sigmoid(self.cag2(enc12, e2))
        a3 = enc23 * torch.sigmoid(self.dag3(enc23, e3)) * torch.sigmoid(self.cag3(enc23, e3))
        a4 = enc34 * torch.sigmoid(self.dag4(enc34, e4)) * torch.sigmoid(self.cag4(enc34, e4))
        
        # Bottleneck
        b = self.bottleneck(a4)
        
        # Decoder
        t12 = self.up4(b)
        d3 = self.dec3(torch.cat([a3, t12], 1))
        t15 = self.up3(d3)
        d2 = self.dec2(torch.cat([a2, t15], 1))
        t17 = self.up2(d2)
        d1 = self.dec1(torch.cat([a1, t17], 1))
        
        # Deep supervision outputs
        o4 = self.out4(b)
        o3 = self.out3(d3)
        o2 = self.out2(d2)
        o1 = self.out1(d1)
        
        # Feature fusion
        t16 = self.fuse_up_t15(t15)
        t14 = self.fuse_up_t12_b(self.fuse_up_t12_a(t12))
        fuse_in = torch.cat([d1, t17, t16, t14], 1)
        fused = self.fuse_conv(fuse_in)
        
        return fused, o1, o2, o3, o4


# ============================================================================
# Utility Functions
# ============================================================================
def load_nifti_as_ras_img(path: str):
    if not path or not os.path.exists(path):
        return None, None
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    data = np.squeeze(img.get_fdata())
    return img, data

def zscore_in_mask(x: np.ndarray, mask: Optional[np.ndarray] = None, eps: float = 1e-6) -> np.ndarray:
    """Z-score normalization within mask region."""
    if mask is None:
        mask = (x != 0)
    vals = x[mask > 0]
    if vals.size < 10:
        m, s = x.mean(), x.std()
    else:
        m, s = float(vals.mean()), float(vals.std())
    s = s if s > eps else eps
    return (x - m) / s


def center_pad(x: np.ndarray, target: Tuple[int, int, int]) -> np.ndarray:
    """Pad array to target shape with centered content."""
    D, H, W = x.shape
    td, th, tw = target
    pad = np.zeros(target, dtype=x.dtype)
    sd = (td - D) // 2
    sh = (th - H) // 2
    sw = (tw - W) // 2
    pad[sd:sd + D, sh:sh + H, sw:sw + W] = x
    return pad


def center_depad(x: np.ndarray, target: Tuple[int, int, int]) -> np.ndarray:
    """Remove padding to restore original shape."""
    D, H, W = x.shape
    td, th, tw = target
    sd = (D - td) // 2
    sh = (H - th) // 2
    sw = (W - tw) // 2
    return x[sd:sd + td, sh:sh + th, sw:sw + tw]


def to_torch(x: np.ndarray) -> torch.Tensor:
    """Convert numpy array to torch tensor."""
    return torch.from_numpy(x.astype(np.float32))


def remove_small_objects_3d(bin_np: np.ndarray, min_size: int) -> np.ndarray:
    """Remove small connected components from binary mask."""
    from scipy.ndimage import label
    lab, num = label(bin_np > 0)
    if num == 0:
        return bin_np
    counts = np.bincount(lab.ravel())
    remove = np.where(counts < min_size)[0]
    out = bin_np.copy()
    for r in remove:
        out[lab == r] = 0
    return out


def postprocess(bin_np: np.ndarray, min_size: int = POST_MIN_SIZE) -> np.ndarray:
    """Apply morphological postprocessing to binary mask."""
    from scipy.ndimage import binary_closing, binary_fill_holes
    
    bin_np = binary_closing(bin_np.astype(bool), structure=np.ones((2, 2, 2))).astype(np.uint8)
    bin_np = binary_fill_holes(bin_np).astype(np.uint8)
    bin_np = remove_small_objects_3d(bin_np, min_size).astype(np.uint8)
    return bin_np


def find_files_path(subject_dir: str, subject_id: str) -> Dict[str, str]:
    """
    Find required input files for a subject (supports both old and BIDS naming).
    Searches in main directory and registration subdirectory.
    
    Args:
        subject_dir: Subject directory path
        subject_id: Subject identifier
    
    Returns:
        Dictionary mapping channel names to file paths
    """
    subject_id_clean = subject_id.replace("sub-", "")
    
    # Search priority: PWI/registration -> DWI/registration -> local fallbacks
    p = Path(subject_dir).expanduser().resolve()
    if p.name == "registration":
        pwi_root = p.parent
    elif p.name == "PWI":
        pwi_root = p
    else:
        pwi_root = p
    subject_root = pwi_root.parent if pwi_root.name in {"PWI", "DWI"} else pwi_root
    search_dirs = [
        str(pwi_root / "registration"),
        str(subject_root / "DWI" / "registration"),
        str(p),
        str(pwi_root),
        str(pwi_root / "segmentation"),
    ]
    
    possible_files = {
        'dwi_mni': [
            # BIDS format (new naming with desc-norm)
            f"{subject_id}_DWI_space-MNI152_aff_desc-norm.nii.gz",
            f"sub-{subject_id_clean}_DWI_space-MNI152_aff_desc-norm.nii.gz",
            f"{subject_id}_space-MNI152_desc-norm_dwi.nii.gz",
            f"sub-{subject_id_clean}_space-MNI152_desc-norm_dwi.nii.gz",
            # BIDS format (standard)
            f"{subject_id}_DWI_space-MNI152_aff.nii.gz",
            f"sub-{subject_id_clean}_DWI_space-MNI152_aff.nii.gz",
            # Old format
            f"{subject_id}_DWI_MNI_Norm.nii.gz",
            f"sub-{subject_id_clean}_DWI_MNI_Norm.nii.gz"
        ],
        'adc_mni': [
            # BIDS format (new naming with desc-norm)
            f"{subject_id}_ADC_space-MNI152_aff_desc-norm.nii.gz",
            f"sub-{subject_id_clean}_ADC_space-MNI152_aff_desc-norm.nii.gz",
            f"{subject_id}_space-MNI152_desc-norm_ADC.nii.gz",
            f"sub-{subject_id_clean}_space-MNI152_desc-norm_ADC.nii.gz",
            # BIDS format (standard)
            f"{subject_id}_ADC_space-MNI152_aff.nii.gz",
            f"sub-{subject_id_clean}_ADC_space-MNI152_aff.nii.gz",
            # Old format
            f"{subject_id}_ADC_MNI_Norm.nii.gz",
            f"sub-{subject_id_clean}_ADC_MNI_Norm.nii.gz"
        ],
        'stroke_mni': [
            # BIDS format (new naming)
            f"{subject_id}_stroke_space-MNI152_aff_desc-norm.nii.gz",
            f"sub-{subject_id_clean}_stroke_space-MNI152_aff_desc-norm.nii.gz",
            # BIDS format (standard)
            f"{subject_id}_space-MNI152_label-stroke_mask.nii.gz",
            f"sub-{subject_id_clean}_space-MNI152_label-stroke_mask.nii.gz",
            f"{subject_id}_stroke_space-MNI152_aff.nii.gz",
            f"sub-{subject_id_clean}_stroke_space-MNI152_aff.nii.gz",
            # Old format
            f"{subject_id}_stroke_registered_aff.nii.gz",
            f"sub-{subject_id_clean}_stroke_registered_aff.nii.gz"
        ],
        'ttp_mni': [
            # BIDS format (new naming with desc-norm)
            f"{subject_id}_TTP_space-MNI152_aff_desc-norm.nii.gz",
            f"sub-{subject_id_clean}_TTP_space-MNI152_aff_desc-norm.nii.gz",
            f"{subject_id}_space-MNI152_desc-norm_TTP.nii.gz",
            f"sub-{subject_id_clean}_space-MNI152_desc-norm_TTP.nii.gz",
            # BIDS format (standard)
            f"{subject_id}_space-MNI152_TTP.nii.gz",
            f"sub-{subject_id_clean}_space-MNI152_TTP.nii.gz",
            f"{subject_id}_TTP_space-MNI152_aff.nii.gz",
            f"sub-{subject_id_clean}_TTP_space-MNI152_aff.nii.gz",
            # Old format
            f"{subject_id}_TTP_brain_aff.nii.gz",
            f"sub-{subject_id_clean}_TTP_brain_aff.nii.gz"
        ],
        'hp_symClassic': [
            # Preferred format
            f"{subject_id}_HPsymclassic-mask_space-MNI152.nii.gz",
            f"sub-{subject_id_clean}_HPsymclassic-mask_space-MNI152.nii.gz",
            # Prior formats
            f"{subject_id}_space-MNI152_label-HP_desc-symClassic_mask.nii.gz",
            f"{subject_id}_space-MNI152_label-HP_desc-symclassic_mask.nii.gz",
            f"sub-{subject_id_clean}_space-MNI152_label-HP_desc-symClassic_mask.nii.gz",
            f"sub-{subject_id_clean}_space-MNI152_label-HP_desc-symclassic_mask.nii.gz",
            # Old format
            f"{subject_id}_HP_symClassic.nii.gz",
            f"sub-{subject_id_clean}_HP_symClassic.nii.gz"
        ],
    }
    
    paths = {}
    for key, filenames in possible_files.items():
        for search_dir in search_dirs:
            if key in paths:  # Already found
                break
            if not os.path.exists(search_dir):
                continue
            for filename in filenames:
                fp = os.path.join(search_dir, filename)
                if os.path.exists(fp):
                    paths[key] = fp
                    break
    
    return paths

def build_channel_list(n_channel: int) -> list:
    """Get required channel names based on number of channels."""
    if n_channel == 3:
        return ["dwi_mni", "adc_mni", "ttp_mni"]
    elif n_channel == 4:
        return ["dwi_mni", "adc_mni", "ttp_mni", "hp_symClassic"]
    elif n_channel == 5:
        return ["dwi_mni", "adc_mni", "stroke_mni", "ttp_mni", "hp_symClassic"]
    else:
        raise ValueError(f"n_channel must be 3, 4, or 5, got {n_channel}")


def autocast_ctx_for(device: torch.device):
    """Get appropriate autocast context for device."""
    return autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


# ============================================================================
# Main Predictor Class
# ============================================================================
class DAGMNetPredictor:
    """DAGMNet predictor with automatic HP_symClassic generation."""
    
    def __init__(
        self,
        checkpoint_path: str,
        n_channels: int = 4,
        threshold: float = DEFAULT_THRESH,
        post_min_size: int = POST_MIN_SIZE,
        device: str = "cuda:0",
        auto_generate_symclassic: bool = True,
        allow_missing_channels: bool = True,
    ):
        self.n_channels = n_channels
        self.threshold = threshold
        self.post_min_size = post_min_size
        self.allow_missing_channels = allow_missing_channels
        
        # ✅ Better device handling
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                print("⚠ CUDA not available, falling back to CPU")
                self.device = torch.device("cpu")
            else:
                try:
                    # Test if the specified device is valid
                    test_device = torch.device(device)
                    torch.zeros(1).to(test_device)  # Test device
                    self.device = test_device
                    print(f"✓ Using device: {self.device}")
                except RuntimeError as e:
                    print(f"⚠ Device {device} not available: {e}")
                    # Try to find an available GPU
                    if torch.cuda.device_count() > 0:
                        self.device = torch.device("cuda:0")
                        print(f"✓ Falling back to: {self.device}")
                    else:
                        self.device = torch.device("cpu")
                        print("✓ Falling back to CPU")
        else:
            self.device = torch.device(device)
        
        self.auto_generate_symclassic = auto_generate_symclassic
        
        # Build model
        print(f"Loading model to {self.device}...")
        self.model = ReplicatedDAGMNet(in_ch=n_channels).to(self.device)
        
        # Load checkpoint
        checkpoint_path = Path(checkpoint_path).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"✓ Loaded DAGMNet checkpoint (epoch {ckpt.get('epoch', '?')})")
        else:
            self.model.load_state_dict(ckpt)
            print("✓ Loaded DAGMNet weights")
        
        self.model.eval()
        print(f"✓ DAGMNet ready on {self.device} with {n_channels} channels")
    
    def _ensure_hp_symclassic(self, subject_dir: str, subject_id: str) -> bool:
        """Ensure HP_symClassic.nii.gz exists, generate if missing."""
        paths = find_files_path(subject_dir, subject_id)
        
        # Check if HP_symClassic already exists
        if 'hp_symClassic' in paths:
            print(f"✓ Found existing HP_symClassic for {subject_id}")
            return True
        
        # If not found and auto-generation is disabled
        if not self.auto_generate_symclassic:
            print(f"⚠ HP_symClassic not found for {subject_id} and auto-generation is disabled")
            return False
        
        # Generate HP_symClassic
        print(f"⚠ HP_symClassic not found for {subject_id}, generating...")
        try:
            generator = HypoperfusionGenerator(subject_dir, subject_id=subject_id)
            hp_map = generator.generate(save_hp=True)
            
            if hp_map is not None:
                print(f"✓ Successfully generated HP_symClassic for {subject_id}")
                return True
            else:
                print(f"❌ Failed to generate HP_symClassic for {subject_id}")
                return False
        except Exception as e:
            print(f"❌ Error generating HP_symClassic for {subject_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    @torch.no_grad()
    def _predict_volume(self, x_full: torch.Tensor) -> np.ndarray:
        """Predict full volume using sliding window with overlapping patches."""
        C, D, H, W = x_full.shape
        pred_canvas = torch.zeros((1, D, H, W), device=self.device)
        count_canvas = torch.zeros((1, D, H, W), device=self.device)
        
        # Sliding window inference with stride
        for x_off in range(DS):
            for y_off in range(DS):
                for z_off in range(Z_STRIDE_FACTOR * DS):
                    # Extract sub-volume
                    sub_x = x_full[:, z_off::(Z_STRIDE_FACTOR * DS), y_off::DS, x_off::DS]
                    
                    # Predict
                    with autocast_ctx_for(self.device):
                        sub_logits, *_ = self.model(sub_x.unsqueeze(0).to(self.device))
                        sub_prob = torch.sigmoid(sub_logits)
                    
                    # Accumulate
                    pred_canvas[:, z_off::(Z_STRIDE_FACTOR * DS), y_off::DS, x_off::DS] += sub_prob[0]
                    count_canvas[:, z_off::(Z_STRIDE_FACTOR * DS), y_off::DS, x_off::DS] += 1.0
        
        # Average overlapping predictions
        pred = torch.where(count_canvas > 0, pred_canvas / count_canvas, 0.0)
        return pred.squeeze(0).squeeze(0).cpu().numpy()
    
    def predict_subject(
        self,
        subject_dir: str,
        subject_id: str,
        save_path: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """
        Predict hypoperfusion mask for a single subject.

        Key invariants enforced here:
        - All channels are loaded in RAS canonical orientation (nib.as_closest_canonical).
        - All channels must share the same voxel grid (shape check).
        - The saved prediction uses a canonical RAS reference (prefer ttp_mni if present),
        so it overlays correctly with TTP in QC.
        """
        warnings.filterwarnings("ignore", category=UserWarning, module="nibabel.nifti1")

        # 0) Determine channels
        channels = build_channel_list(self.n_channels)

        # 1) Resolve available channel files (and optionally generate hp_symClassic)
        paths = find_files_path(subject_dir, subject_id)
        if "hp_symClassic" in channels and "hp_symClassic" not in paths:
            hp_ready = self._ensure_hp_symclassic(subject_dir, subject_id)
            if hp_ready:
                paths = find_files_path(subject_dir, subject_id)
            elif not self.allow_missing_channels:
                print(f"❌ Cannot proceed without HP_symClassic for {subject_id}")
                return None
            else:
                print(f"⚠ Proceeding without HP_symClassic for {subject_id}; channel will be zero-filled.")

        missing = [ch for ch in channels if ch not in paths]
        if missing and not self.allow_missing_channels:
            print(f"❌ Missing required files for {subject_id}: {missing}")
            print("\n💡 Available files:")
            for key, p in paths.items():
                print(f"   ✓ {key}: {os.path.basename(p)}")
            return None
        if missing and self.allow_missing_channels:
            print(f"⚠ Missing channels for {subject_id}: {missing} (will zero-fill)")

        print(f"Processing {subject_id} with channel order: {channels}")

        # 2) Load available channels as RAS canonical + enforce same grid
        arrays: Dict[str, np.ndarray] = {}
        imgs: Dict[str, nib.Nifti1Image] = {}

        for ch in channels:
            p = paths.get(ch)
            if p is None:
                continue
            img, arr = load_nifti_as_ras_img(p)  # must canonicalize to RAS
            if img is None or arr is None:
                if self.allow_missing_channels:
                    print(f"⚠ Failed to load {ch} for {subject_id}; channel will be zero-filled.")
                    continue
                print(f"❌ Failed to load {ch} for {subject_id}")
                return None
            imgs[ch] = img
            arrays[ch] = arr
        if not arrays:
            print(f"❌ No valid input channels found for {subject_id}")
            return None

        shapes = {ch: arrays[ch].shape for ch in arrays}
        if len(set(shapes.values())) != 1:
            raise RuntimeError(f"Channel grid mismatch for {subject_id}: {shapes}")
        ref_channel = next(ch for ch in channels if ch in arrays)

        # 3) Build brain mask from existing channels (non-zero union + finite)
        brain_mask = None
        for ch in arrays:
            arr = arrays[ch]
            valid = np.isfinite(arr)
            nonzero = (arr != 0) & valid
            brain_mask = nonzero if brain_mask is None else (brain_mask | nonzero)

        if brain_mask is None or int(brain_mask.sum()) < 10:
            brain_mask = np.isfinite(arrays[ref_channel])

        # 4) Normalize/pad existing channels; missing channels are zero tensors.
        padded_arrays = {}
        for ch in arrays:
            x = arrays[ch]
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            x = zscore_in_mask(x, brain_mask)
            x = center_pad(x, PAD_SHAPE)
            padded_arrays[ch] = x.astype(np.float32, copy=False)

        # 5) Stack channels in fixed training order [C, D, H, W].
        x_full_np = np.stack(
            [
                padded_arrays[ch] if ch in padded_arrays else np.zeros(PAD_SHAPE, dtype=np.float32)
                for ch in channels
            ],
            axis=0,
        )
        x_full_t = to_torch(x_full_np)

        print(f"Input shape: {tuple(x_full_t.shape)}")

        prob_pad = self._predict_volume(x_full_t)

        # 6) Depad to original volume size inferred from first available channel.
        ref_shape = arrays[ref_channel].shape
        prob_mni = center_depad(prob_pad, ref_shape)

        # 7) Threshold and postprocess
        pred_bin = (prob_mni > float(self.threshold)).astype(np.uint8)
        pred_bin = postprocess(pred_bin, min_size=int(self.post_min_size))

        print(f"✓ Prediction complete. Volume: {int(pred_bin.sum())} voxels")

        # 8) Save (prefer TTP header if available; fallback to first loaded channel)
        if save_path:
            ref_key = "ttp_mni" if "ttp_mni" in imgs else ref_channel
            ref_img = imgs[ref_key]  # already canonical RAS

            out = nib.Nifti1Image(pred_bin.astype(np.float32), ref_img.affine, ref_img.header)
            # keep header consistent and simple
            out.set_data_dtype(np.float32)
            nib.save(out, save_path)
            print(f"✓ Saved prediction to: {save_path}")

            # Optional debug (uncomment if needed)
            # import nibabel as nib
            # print("Ref axcodes:", nib.aff2axcodes(ref_img.affine))
            # print("Out axcodes:", nib.aff2axcodes(out.affine))

        return pred_bin



# ============================================================================
# Standalone Usage
# ============================================================================
def main():
    """Example standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DAGMNet single-subject inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--subject_dir", type=str, required=True, help="Subject directory")
    parser.add_argument("--subject_id", type=str, required=True, help="Subject ID")
    parser.add_argument("--n_channels", type=int, default=4, choices=[3, 4, 5])
    parser.add_argument("--output", type=str, default=None, help="Output path for prediction")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESH)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no_auto_symclassic", action="store_true", 
                       help="Disable automatic HP_symClassic generation")
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = DAGMNetPredictor(
        checkpoint_path=args.checkpoint,
        n_channels=args.n_channels,
        threshold=args.threshold,
        device=args.device,
        auto_generate_symclassic=not args.no_auto_symclassic
    )
    
    # Run prediction
    pred_mask = predictor.predict_subject(
        subject_dir=args.subject_dir,
        subject_id=args.subject_id,
        save_path=args.output
    )
    
    if pred_mask is not None:
        print(f"\n✓ Success! Predicted {pred_mask.sum()} voxels")
    else:
        print("\n❌ Prediction failed")


if __name__ == "__main__":
    main()
