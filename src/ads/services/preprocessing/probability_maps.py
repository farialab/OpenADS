"""
Stroke probability map calculation service.
"""

from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Union, Optional

import numpy as np
import ants
import scipy.special
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


class ProbabilityMapCalculator:
    """Calculates probability maps for ischemic stroke."""

    def __init__(
        self,
        model_vars: Optional[list[float]] = None
    ):
        """Initialize probability map calculator.

        Args:
            model_vars: Model parameters [fwhm, alpha_dwi, lambda_dwi,
                       alpha_adc, lambda_adc, id_isch_zth]
                       Default: [2, 1.5, 4, 0.5, 2, 2]
        """
        self.model_vars = model_vars or [2, 1.5, 4, 0.5, 2, 2]

    def compute(
        self,
        dwi_norm: np.ndarray,
        adc_img: np.ndarray,
        mask_img: np.ndarray,
        template_dir: Union[Path, str]
    ) -> np.ndarray:
        """Calculate probability map for ischemic stroke.

        Args:
            dwi_norm: Normalized DWI image in MNI space
            adc_img: ADC image in MNI space
            mask_img: Brain mask in MNI space
            template_dir: Directory containing template images

        Returns:
            Probability map for ischemic stroke

        Note:
            Equivalent to get_stroke_probability_map() from preprocessing.py
        """
        # Convert ANTs images to numpy if needed
        if isinstance(dwi_norm, ants.ANTsImage):
            dwi_norm = dwi_norm.numpy()
        if isinstance(adc_img, ants.ANTsImage):
            adc_img = adc_img.numpy()
        if isinstance(mask_img, ants.ANTsImage):
            mask_img = mask_img.numpy() > 0.49

        # Load templates
        templates = self._load_templates(template_dir)

        # Unpack model variables
        fwhm, alpha_dwi, lambda_dwi, alpha_adc, lambda_adc, id_isch_zth = self.model_vars

        # Calculate DWI dissimilarity
        dwi_h2 = self._process_dissimilarity(
            dwi_norm,
            templates['dwi_mu'],
            templates['dwi_std'],
            alpha_dwi,
            lambda_dwi,
            is_dwi=True
        )
        dwi_h2[dwi_norm < id_isch_zth] = 0
        dwi_h2 = dwi_h2 * dwi_norm

        # Calculate ADC dissimilarity
        adc_h1 = self._process_dissimilarity(
            adc_img,
            templates['adc_mu'],
            templates['adc_std'],
            alpha_adc,
            lambda_adc,
            is_dwi=False
        )
        adc_h1 = adc_h1 * mask_img

        # Calculate ischemic indicator
        id_isch = (1 - self._qfunc(dwi_norm/id_isch_zth)) * (dwi_norm > id_isch_zth)

        # Calculate final probability map
        prob_is = dwi_h2 * adc_h1 * id_isch * mask_img

        return prob_is

    def _load_templates(self, template_dir: Union[Path, str]) -> dict[str, np.ndarray]:
        """Load template images.

        Args:
            template_dir: Directory containing templates

        Returns:
            Dictionary of template arrays
        """
        template_paths = {
            'dwi_mu': os.path.join(template_dir, 'normal_mu_dwi_Res_ss_MNI_scaled_normalized.nii.gz'),
            'dwi_std': os.path.join(template_dir, 'normal_std_dwi_Res_ss_MNI_scaled_normalized.nii.gz'),
            'adc_mu': os.path.join(template_dir, 'normal_mu_ADC_Res_ss_MNI_normalized.nii.gz'),
            'adc_std': os.path.join(template_dir, 'normal_std_ADC_Res_ss_MNI_normalized.nii.gz')
        }

        return {k: ants.image_read(v).numpy() for k, v in template_paths.items()}

    def _process_dissimilarity(
        self,
        img_data: np.ndarray,
        template_mu: np.ndarray,
        template_std: np.ndarray,
        alpha: float,
        lambda_val: float,
        is_dwi: bool
    ) -> np.ndarray:
        """Calculate dissimilarity map.

        Args:
            img_data: Input image data
            template_mu: Template mean
            template_std: Template std deviation
            alpha: Scaling parameter
            lambda_val: Power parameter
            is_dwi: True for DWI processing, False for ADC

        Returns:
            Dissimilarity map
        """
        # Normalize input
        img_norm = (img_data - np.mean(img_data)) / np.std(img_data)

        # Apply Gaussian filter
        g_sigma = self.model_vars[0] / 2 / np.sqrt(2 * np.log(2))
        img_smooth = self._apply_gaussian_filter(img_norm, g_sigma)

        # Calculate dissimilarity
        dissim = np.tanh((img_smooth - template_mu) / template_std / alpha)

        if is_dwi:
            dissim[dissim < 0] = 0
            return dissim ** lambda_val
        else:
            dissim[dissim > 0] = 0
            return (-dissim) ** lambda_val

    def _apply_gaussian_filter(self, img_data: np.ndarray, sigma: float) -> np.ndarray:
        """Apply 3D Gaussian filter slice by slice.

        Args:
            img_data: Input image
            sigma: Gaussian sigma

        Returns:
            Filtered image
        """
        filtered = np.zeros_like(img_data)
        for i in range(img_data.shape[-1]):
            filtered[:, :, i] = gaussian_filter(img_data[:, :, i], sigma)
        return filtered

    @staticmethod
    def _qfunc(x: np.ndarray) -> np.ndarray:
        """Q-function (tail probability of normal distribution).

        Args:
            x: Input values

        Returns:
            Q-function values
        """
        return 0.5 - 0.5 * scipy.special.erf(x / np.sqrt(2))
