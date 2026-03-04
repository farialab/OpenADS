# --- Imports ---
import os
import sys
import json
import time
import tempfile
import traceback
import multiprocessing as mp
from pathlib import Path
import numpy as np
import nibabel as nib
import ants
from scipy.ndimage import (
    gaussian_filter,
    binary_fill_holes,
    binary_erosion
)
from scipy.signal import find_peaks
from scipy.interpolate import InterpolatedUnivariateSpline as Interp


# Add parent directory to path for imports
os.sys.path.append(str(Path(__file__).resolve().parents[3]))
os.sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))

from ads.core.brainmask import generate_brain_mask_with_synthstrip

#!/usr/bin/env python3
"""
pwi_to_ttp_synth.py

Preprocess PWI data to a 3D TTP map using SynthStrip for skull-stripping.
"""
########################################################Configuration#########################################
class PWIPreprocessingConfig:
    def __init__(self):
        # Configuration for correcting slice timing issues in PWI data
        self.slicetime_correction = {
            'slice_order_code': 2,  # Interleaved ascending (even-first)
            'interp_degree': 1,  # Degree for linear interpolation (spline degree must be between 1 and 5)
        }

        # Configuration for motion correction processes
        self.motion_correction = {
            'motion_correction': True,  # Enable PWI motion correction
            'save_corrected_PWI': False,  # Save corrected PWI images if True (default is False)
            'registration_params': {  # Parameters for rigid body registration
                'mode': 'rigid',
                'level_iters': [100, 10],  # Iteration levels for the registration process
                'sigmas': [3.0, 1.0],  # Sigma values for the registration algorithm
                'factors': [4, 2]  # Scaling factors used in the registration process
            }
        }

        # Configuration for brain masking using median filtering and otsu thresholding
        self.brain_masking_method = 'synthstrip'  # options: 'otsu', 'synthstrip'
        self.brain_masking_config = {
            'otsu': {
                'median_radius': 2,
                'numpass': 4,
                'min_size_remove_objects': 100
            },
            'synthstrip': {
                'save_ss': False,
                'no_csf': False
            }
        }

        # Configuration for Gadolinium concentration calculation and baseline adjustments
        self.gd_concentration = {
            'gauss_filtering': {
                'fwhm_spatial': 2.35,  # Full-width at half-maximum for spatial filtering
                'fwhm_temporal': 3.53,  # Full-width at half-maximum for temporal filtering
            },
            'g0_baseline': {
                'baseline_volumes': 6, # Number of baseline_volumes used to compute baseline Gd
                'adaptive_baseline': True,  # Dynamically adjust baseline based on the percentage of slices
                'baseline_percentage': 0.10,  # At least 10% of slices used to compute baseline Gd
                'skip_volumes': 1,  # Volumes to skip at the start for baseline calculation
                'trim_high': 1,  # Exclude high outliers in baseline calculation
                'trim_low': 1   # Exclude low outliers in baseline calculation
            },
            'gd_calculation': {
                'echo_time': 0.03,  # Echo time used in the calculation
            }
        }

        # Parameters for peak detection in Gd concentration data
        self.ttp_peaks = {
            'num_peaks': 3,
            'prominence': 1,
            'height': 2,
            'width': 3,
        }

        self.adc2ttp_regParams = {
            'mode': 'affine',
            'level_iters': [1000, 1000, 1000],
            'sigmas': [6.0, 4.0, 2.0],
            'factors': [4, 2, 2]
        }

    def update_config(self, category, **kwargs):
        """
        Update the configuration within a specific category with provided keyword arguments.
        """
        if category in self.__dict__:
            self.__dict__[category].update(kwargs)
        else:
            print(f"Warning: {category} is not a valid category")

    def update_baseline_config(self, total_volumes):
        """
        Dynamically adjust baseline volumes based on the total number of volumes and a specified percentage.
        """
        if self.gd_concentration['g0_baseline']['adaptive_baseline']:
            min_volumes = int(total_volumes * self.gd_concentration['g0_baseline']['baseline_percentage'])
            current_volumes = self.gd_concentration['g0_baseline']['baseline_volumes']
            self.gd_concentration['g0_baseline']['baseline_volumes'] = max(min_volumes, current_volumes)
            # Correct the comment here to reflect the action
            self.ttp_peaks['skip_volumes'] = self.gd_concentration['g0_baseline']['baseline_volumes'] + self.gd_concentration['g0_baseline']['skip_volumes']

    def display(self):
        """
        Display the current configuration organized by category.
        """
        for category, config_dict in self.__dict__.items():
            print(f"Category: {category}")
            for key, value in config_dict.items():
                print(f"  {key}: {value}")
            print()  # Additional newline for clearer separation


##########################################
from scipy.ndimage import morphology, gaussian_filter
import skimage.morphology as skimorph
import scipy

def generate_pwi_mc_brain_mask(pwi_mc_path, output_dir, method='synthstrip'):
    """
    Generates a brain mask for motion-corrected PWI using 'synthstrip' or 'otsu'.
    """
    subject_id = os.path.basename(pwi_mc_path).split('_')[0]
    if method == 'synthstrip':
        # create a temp dir (pwi_mc_brain_mask_temp) under output_path
        temp_path = os.path.join(output_dir, "pwi_mc_brain_mask_temp")
        os.makedirs(temp_path, exist_ok=True)

        pwi_mc_img = nib.load(pwi_mc_path).get_fdata()
        mask = np.ones_like(pwi_mc_img[:,:,:,0])

        for i in range(0,pwi_mc_img.shape[-1],5):
            pwi_mc_t = pwi_mc_img[:,:,:,i] #save it
            pwi_mc_t_path = os.path.join(temp_path, f"{subject_id}_pwi_mc_t_{i}.nii.gz")
            #nib.save(nib.Nifti1Image(pwi_mc_t, pwi_mc_img.affine, pwi_mc_img.header), pwi_mc_t_path)
            nib.save(nib.Nifti1Image(pwi_mc_t, nib.load(pwi_mc_path).affine, nib.load(pwi_mc_path).header), pwi_mc_t_path)

            mask_path = os.path.join(temp_path, f"{subject_id}_synthstrip_brain_mask_raw_{i}.nii.gz")

            mask_ = generate_brain_mask_with_synthstrip(pwi_mc_t_path, mask_path,
                use_gpu=True,
                no_csf=False,
                model_path=None
            )
            print(f"mask.shape != mask_.shape: {mask.shape} != {mask_.shape}")
            if mask.shape != mask_.shape:
                raise ValueError(f"Mask shape mismatch: {mask.shape} != {mask_.shape}")
                mask = np.ones_like(mask_)
            mask = mask*mask_
    else:
        raise ValueError("Invalid masking method. Choose 'synthstrip'.")

    for i in range(mask.shape[2]):
        mask[:,:,i] = morphology.binary_fill_holes(mask[:,:,i]>0.5)  
        mask[:,:,i] = morphology.binary_erosion(mask[:,:,i])
        mask[:,:,i] = skimorph.remove_small_objects(mask[:,:,i]>0.5, min_size=100)
        mask_eroded = morphology.binary_erosion(mask[:,:,i], iterations=2)
        mask[:,:,i] = scipy.ndimage.binary_propagation(mask_eroded, mask=mask[:,:,i])
        
    mask_eroded = morphology.binary_erosion(mask, iterations=int(mask.shape[-1]/3))
    mask = scipy.ndimage.binary_propagation(mask_eroded, mask=mask)

    # save the mask
    mask_nib = nib.Nifti1Image(mask.astype(np.float32), affine=nib.load(pwi_mc_path).affine, header=nib.load(pwi_mc_path).header)

    nib.save(mask_nib, os.path.join(output_dir, f"{subject_id}_PWI_MC_BrainMask_{method}.nii.gz"))

    return mask

# processing
#########################################################slicetime_correction#######################################
import os
import sys
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as Interp

def get_repetition_time(PWI_json, PWI_imgJ):
    """
    Extract the repetition time from JSON or NIFTI header. If the repetition time is 0, set it to 1.6.
    """
    if 'RepetitionTime' in PWI_json and PWI_json['RepetitionTime'] not in (None, 0):
        repetition_time = PWI_json['RepetitionTime']
    else:
        print('--- No valid RepetitionTime in PWI.json, using default TR=1.7')
        repetition_time = 1.7

    return repetition_time

def validate_parameters(TR, N_slices_z):
    """
    Validate repetition time and slice count.
    """
    if TR == 0:
        raise ValueError('RepetitionTime is invalid!')
    if N_slices_z == 0:
        raise ValueError('Image has invalid fourth dimension!')

def determine_slice_timing(PWI_json, TR, N_slices_z, slice_order_code):
    """
    Determine slice order and timing based on available data or predefined order.
    """
    if 'SliceTiming' in PWI_json and len(set(PWI_json['SliceTiming'])) == len(PWI_json['SliceTiming']):
        slice_timing = np.array(PWI_json['SliceTiming'])
        order_idx = np.argsort(slice_timing).astype(int)
    else:
        print('--- Defaulting slice_timing based on code', slice_order_code)
        order_idx, slice_timing = default_slice_timing(N_slices_z, TR, slice_order_code)
    return order_idx, slice_timing

def default_slice_timing(N_slices_z, TR, slice_order_code):
    """
    Set default slice timing based on the order code.
    """
    step = TR / N_slices_z
    if slice_order_code == 1:
        return np.arange(N_slices_z), np.arange(0, TR, step)
    elif slice_order_code == 2:
        return np.array(list(range(0, N_slices_z, 2)) + list(range(1, N_slices_z, 2))), np.arange(0, TR, step)
    else:
        return np.array(list(range(1, N_slices_z, 2)) + list(range(0, N_slices_z, 2))), np.arange(0, TR, step)

def slicetime_correction(PWI_imgJ, PWI_img, PWI_json, slice_order_code=3, interp_degree=1, epi = 1E-6):
    TR = get_repetition_time(PWI_json, PWI_imgJ)
    N_slices_z = PWI_img.shape[2]
    validate_parameters(TR, N_slices_z)
    order_idx, slice_timing = determine_slice_timing(PWI_json, TR, N_slices_z, slice_order_code)

    time_ref = np.arange(np.min(slice_timing), TR * PWI_img.shape[-1], TR)[:PWI_img.shape[-1]]
    time_slices = [np.arange(np.min(slice_timing) + slice_timing[idx], TR * PWI_img.shape[-1], TR)[:PWI_img.shape[-1]] for idx in order_idx]

    PWI_corrected = np.zeros_like(PWI_img)
    for iz in range(N_slices_z):
        for ix in range(PWI_img.shape[0]):
            for iy in range(PWI_img.shape[1]):
                PWIv = PWI_img[ix, iy, iz, :]
                interper = Interp(time_slices[iz], PWIv, k=interp_degree)
                PWI_corrected[ix, iy, iz, :] = interper(time_ref)
    PWI_corrected[PWI_corrected <= 0] = epi
    return PWI_corrected, time_slices, time_ref

#########################################################slicetime_correction#######################################

#########################################################Motion correction#######################################

def apply_precomputed_motion_correction(img, affines_dir_path):
    """
    Apply precomputed motion correction affines to a 4D image array.
    """
    print("Using pre-computed motion correction affines.")
    img_unmerged = ants.ndimage_to_list(img)
    motion_corrected = [img_unmerged[0]]
    for i in range(1, len(img_unmerged)):
        transform_path = os.path.join(affines_dir_path, f"motion_correction_tx_{i}.mat")
        motion_corrected.append(ants.apply_transforms(img_unmerged[0], img_unmerged[i], transform_path))
    return ants.list_to_ndimage(img, motion_corrected)

def motion_correction(img, **reg_params):
    """
    Apply motion correction across all volumes of a 4D image array using specified registration parameters.
    """
    img_unmerged = ants.ndimage_to_list(img)

    motion_corrected = list()
    reg_corrected = list()
    for i in range(1, len( img_unmerged ) ):
        tx = ants.registration( img_unmerged[0], img_unmerged[i], 
                                type_of_transform="Affine",
                                aff_iterations=(1000, 500, 250, 100),
                                metric="mattes",
                                metric_radius=32,
                                verbose=False
                               )
        reg_corrected.extend(tx["fwdtransforms"])
        motion_corrected.append( tx["warpedmovout"] )
    
    motion_corrected_img = ants.list_to_ndimage(img, motion_corrected)
    return motion_corrected_img, reg_corrected


def _motion_correction_worker(img_path, reg_params, affines_dir_path, corrected_img_path, error_path):
    """Run motion correction in a subprocess so the parent can enforce a timeout."""
    try:
        img = ants.image_read(str(img_path))
        corrected_image, mc_reg_affines = motion_correction(img, **reg_params)

        os.makedirs(affines_dir_path, exist_ok=True)
        for i, transform in enumerate(mc_reg_affines):
            transform_path = os.path.join(affines_dir_path, f"motion_correction_tx_{i+1}.mat")
            ants.write_transform(ants.read_transform(transform), transform_path)

        ants.image_write(corrected_image, corrected_img_path)
    except Exception:
        with open(error_path, "w") as f:
            f.write(traceback.format_exc())


def handle_motion_correction(img_path, affines_dir_path, settings):
    """
    Handle motion correction for a 4D image either by applying pre-computed affines
    or by performing new motion correction.
    """
    img = ants.image_read(img_path)
    if all(os.path.exists(os.path.join(affines_dir_path, f"motion_correction_tx_{i+1}.mat")) for i in range(img.shape[-1] - 1)):
        corrected_image = apply_precomputed_motion_correction(img, affines_dir_path)
    else:
        print("Performing motion correction.")
        timeout_seconds = settings.motion_correction.get("timeout_seconds", None)
        skip_if_timeout = bool(settings.motion_correction.get("skip_if_timeout", True))

        # Keep old behavior when timeout is not configured.
        if timeout_seconds is None:
            os.makedirs(affines_dir_path, exist_ok=True)
            corrected_image, MC_reg_affines = motion_correction(img, **settings.motion_correction['registration_params'])
            for i, transform in enumerate(MC_reg_affines):
                transform_path = os.path.join(affines_dir_path, f"motion_correction_tx_{i+1}.mat")
                ants.write_transform(ants.read_transform(transform), transform_path)
        else:
            with tempfile.TemporaryDirectory(prefix="pwi_moco_") as tmpdir:
                corrected_img_path = os.path.join(tmpdir, "corrected.nii.gz")
                error_path = os.path.join(tmpdir, "worker_error.txt")

                proc = mp.Process(
                    target=_motion_correction_worker,
                    args=(
                        str(img_path),
                        settings.motion_correction['registration_params'],
                        str(affines_dir_path),
                        corrected_img_path,
                        error_path,
                    ),
                    daemon=True,
                )
                proc.start()
                proc.join(float(timeout_seconds))

                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)
                    msg = (
                        f"Motion correction timed out after {timeout_seconds}s for {img_path}. "
                        "Skipping motion correction and continuing with slice-time corrected PWI."
                    )
                    if skip_if_timeout:
                        print(f"[WARN] {msg}")
                        return img.numpy()
                    raise TimeoutError(msg)

                if proc.exitcode != 0 and os.path.exists(error_path):
                    with open(error_path, "r") as f:
                        worker_trace = f.read()
                    raise RuntimeError(f"Motion correction subprocess failed:\n{worker_trace}")

                if not os.path.exists(corrected_img_path):
                    raise RuntimeError("Motion correction subprocess finished without output image.")

                corrected_image = ants.image_read(corrected_img_path)
    return corrected_image.numpy()

#########################################################Motion correction#######################################

#########################################################get_gd_concentration#####################################
from scipy import signal as ssi
from scipy import ndimage as ndi
import math
import numpy as np 

def get_gkern1d(fwhm=3, p=1):
    g_sigma = fwhm/2/np.sqrt(2*np.log(2))
    kernlen = math.ceil(7*g_sigma)
    """Returns a 3D Gaussian kernel array."""
    gkern1d = ssi.general_gaussian(kernlen, sig=g_sigma, p=p)
    return gkern1d

def gaussian_filter_image(image, voxel_dims, fwhm_spatial, fwhm_temporal):
    scaling = 2.*np.sqrt(2.*np.log(2.))
    safe_xyz = [vox if np.isfinite(vox) and vox > 0 else 1.0 for vox in voxel_dims[:3]]
    safe_t = voxel_dims[3] if len(voxel_dims) > 3 and np.isfinite(voxel_dims[3]) and voxel_dims[3] > 0 else 1.0
    sigma = [fwhm_spatial / vox / scaling for vox in safe_xyz]
    sigma.append(fwhm_temporal / safe_t / scaling)
    return ndi.gaussian_filter(image, sigma)

def compute_baseline_intensity(image, baseline_volumes, skip_volumes, trim_low, trim_high, **kwarg):

    if trim_low + trim_high >= baseline_volumes:
        raise ValueError('trim_high + trim_low must be less than the number of baseline volumes')
    
    baseline_image = image[..., skip_volumes:skip_volumes + baseline_volumes]
    baseline_image = np.sort(baseline_image, axis=-1)
    baseline_image = baseline_image[..., trim_low:-trim_high]
    return np.mean(baseline_image, axis=-1)

def compute_gd_concentration(image, baseline_intensity, echo_time, epsilon = 1E-6):
    concentration = -(1 / echo_time) * np.log(np.maximum(image / (baseline_intensity + epsilon), epsilon))
    return concentration

def compute_gd_map(image, mask, voxel_dims, settings):
    filtered_image = gaussian_filter_image(image, voxel_dims, **settings['gauss_filtering'])

    baseline_intensity = compute_baseline_intensity(filtered_image, **settings['g0_baseline'])
    baseline_intensity[mask < 0.5] = np.min(baseline_intensity[mask > 0.5])

    gd_concentration_map = np.zeros_like(image)
    for i in range(image.shape[-1]):
        gd_concentration_map[..., i] = compute_gd_concentration(image[..., i], baseline_intensity, **settings['gd_calculation'])
    
    gd_concentration_map[mask < 0.5] = np.min(gd_concentration_map[mask > 0.5])
    return gd_concentration_map, baseline_intensity


##########################################

#########################################################calculate_ttp_peaks#####################################
# computeTTP.py
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import binary_dilation
import scipy.interpolate
from scipy.stats import mode

from joblib import Parallel, delayed


def find_signal_peaks(signal, ts, prominence, height, width, skip_volumes, num_peaks):
    peaks, _ = find_peaks(signal, prominence=prominence, height=height, width=width)
    valid_peaks = peaks[peaks > skip_volumes]
    
    if len(valid_peaks) < num_peaks:
        if valid_peaks.size > 0:
            valid_peaks = np.pad(valid_peaks, (0, num_peaks - len(valid_peaks)), 'constant', constant_values=valid_peaks[-1])
        else:
            return np.full(num_peaks, np.nan)
    return ts[valid_peaks][:num_peaks]

def compute_single_voxel_peak(ix, iy, iz, gd_image, ts, peak_settings):
    signal = gd_image[ix, iy, iz, :]
    return find_signal_peaks(signal, ts, **peak_settings)

def compute_peak_ttp_map_parallel(gd_image, mask, voxel_dims, peak_settings):
    ts = np.arange(gd_image.shape[-1]) * voxel_dims[-1]
    peak_ttp_map = np.full(gd_image.shape[:-1] + (peak_settings['num_peaks'],), np.nan)

    indices = np.argwhere(mask > 0.5)
    results = Parallel(n_jobs=-1)(delayed(compute_single_voxel_peak)(ix, iy, iz, gd_image, ts, peak_settings) 
                                  for ix, iy, iz in indices)
    results_array = np.array(results)  # Shape (len(indices), num_peaks)
    peak_ttp_map[indices[:, 0], indices[:, 1], indices[:, 2], :] = results_array
    return peak_ttp_map


def fill_nan_with_neighbors(TTP_img, peak_ttp_map, mask, neighborhood_size):
    """
    Propagate non-NaN values into NaN areas based on neighborhood information, focusing on matching peak values.

    Args:
        TTP_img (numpy.ndarray): 3D array with initial TTP values and NaNs.
        peak_ttp_map (numpy.ndarray): 4D array where the last dimension contains the peak times.
        mask (numpy.ndarray): Binary mask for valid regions.
        neighborhood_size (tuple): Size of the neighborhood for considering propagation (e.g., (3, 3, 1)).
    
    Returns:
        numpy.ndarray: 3D array with the propagated or adjusted TTP values from neighbors.
    """
    struct = np.ones(neighborhood_size, dtype=bool)
    TTP_img[mask < 0.5] = 0
    nan_mask = np.isnan(TTP_img) & (mask > 0.5)
    previous_unfilled_count = np.sum(nan_mask)
    
    while np.any(nan_mask):
        expanded_mask = binary_dilation(~nan_mask & (mask > 0.5), structure=struct) & nan_mask
        fill_indices = np.argwhere(expanded_mask)
        
        for ix, iy, iz in fill_indices:
            local_slice = (
                slice(max(ix - neighborhood_size[0], 0), min(ix + neighborhood_size[0] + 1, TTP_img.shape[0])),
                slice(max(iy - neighborhood_size[1], 0), min(iy + neighborhood_size[1] + 1, TTP_img.shape[1])),
                slice(max(iz - neighborhood_size[2], 0), min(iz + neighborhood_size[2] + 1, TTP_img.shape[2]))
            )

            candidate_peaks = peak_ttp_map[local_slice][..., 0]  # Get the first peak for simplicity in this slice
            neighbor_peaks = TTP_img[local_slice]
            valid_peaks = candidate_peaks[~np.isnan(candidate_peaks) & ~np.isnan(neighbor_peaks)]
            
            if valid_peaks.size > struct.size / 3:
                mode_result = mode(valid_peaks)
                mode_valid_peak = mode_result.mode
                target_peaks = peak_ttp_map[ix, iy, iz, :]
                # Check if any peak is non-NaN and calculate differences
                if np.any(~np.isnan(target_peaks)):
                    differences = np.abs(target_peaks - mode_valid_peak)
                    min_peak = target_peaks[np.nanargmin(differences)]
                else:
                    min_peak = mode_valid_peak
                TTP_img[ix, iy, iz] = min_peak
                nan_mask[ix, iy, iz] = False

        current_unfilled_count = np.sum(nan_mask)
        # Check if no new NaNs were filled in the last iteration
        if current_unfilled_count == previous_unfilled_count:
            break
        previous_unfilled_count = current_unfilled_count
    
    TTP_img[mask < 0.5] = 0
    return TTP_img


def denoise_ttp(ttp_image, brain_mask, voxel_dims, z_thresholds=(3, 3), quantile_thresholds=(0.99, 0.0005)):
    """
    Corrects noise in a TTP image using the provided brain mask and voxel dimensions.
    
    Parameters:
    ttp_image (np.ndarray): The input TTP image.
    brain_mask (np.ndarray): The mask indicating brain regions.
    voxel_dims (tuple): The dimensions of each voxel (dx, dy, dz).
    z_thresholds (tuple): The upper and lower z-score thresholds for valid brain regions.
    quantile_thresholds (tuple): The upper and lower quantile thresholds for valid brain regions.
    
    Returns:
    np.ndarray: The corrected TTP image.
    """
    
    # Apply the brain mask to select valid brain regions
    brain_mask = (brain_mask > 0.5).astype(float)
    valid_brain_mask = (~np.isnan(ttp_image)) & (brain_mask > 0.5)
    
    # Extract voxel dimensions and override dz
    dx, dy, dz = voxel_dims    
    # Calculate mean and standard deviation of valid brain regions
    valid_ttp_values = ttp_image[valid_brain_mask]
    mean_ttp = np.mean(valid_ttp_values)
    std_ttp = np.std(valid_ttp_values)
    
    # Extract thresholds
    upper_z_threshold, lower_z_threshold = z_thresholds
    upper_quantile_threshold, lower_quantile_threshold = quantile_thresholds
    
    # Apply quantile and z-score thresholds to filter valid brain regions
    upper_quantile = np.quantile(valid_ttp_values, upper_quantile_threshold)
    lower_quantile = np.quantile(valid_ttp_values, lower_quantile_threshold)
    valid_brain_mask &= (ttp_image < (upper_quantile + 0.01)) & (ttp_image > (lower_quantile - 0.01))
    valid_brain_mask &= (ttp_image < (mean_ttp + upper_z_threshold * std_ttp)) & (ttp_image > (mean_ttp - lower_z_threshold * std_ttp))
    
    # Create a meshgrid for interpolation
    xx, yy, zz = np.meshgrid(
        np.arange(ttp_image.shape[0]),
        np.arange(ttp_image.shape[1]),
        np.arange(ttp_image.shape[2]),
        indexing='ij'
    )
    
    # Prepare data for interpolation
    valid_coordinates = np.vstack((xx[valid_brain_mask] * dx, yy[valid_brain_mask] * dy, zz[valid_brain_mask] * dz)).T
    valid_ttp_values = ttp_image[valid_brain_mask]
    
    # Perform nearest neighbor interpolation
    interpolator = scipy.interpolate.NearestNDInterpolator(valid_coordinates, valid_ttp_values)
    interpolated_ttp_image = interpolator(xx * dx, yy * dy, zz * dz).reshape(xx.shape)
    
    # Apply the brain mask to the result
    interpolated_ttp_image[brain_mask < 0.5] = 0
    
    return interpolated_ttp_image

def compute_ttp(peak_ttp_map, mask, voxel_dims, neighborhood_size = (3, 3, 1), z_thresholds=(3, 3), quantile_thresholds=(0.99, 0.0005)):
    """
    Generates an image of the first peak time if peaks are close, and propagates neighboring values into NaN voxels and denoise ttp.


    Args:
    peak_ttp_map (numpy.ndarray): 4D array where last dimension contains the peak times.
    mask (numpy.ndarray): Binary mask indicating valid voxels for analysis.
    neighborhood_size (tuple): Size of the neighborhood for considering propagation (e.g., (3, 3, 1)).
    z_thresholds (tuple): The upper and lower z-score thresholds for valid brain regions.
    quantile_thresholds (tuple): The upper and lower quantile thresholds for valid brain regions.
    
    Returns:
    numpy.ndarray: 3D array with the first TTP or propagated values from neighbors.
    """
    # Initialize the result image with NaNs
    TTP_img = np.full(peak_ttp_map.shape[:-1], np.nan)
    valid_mask = mask > 0.5
    TTP_img[valid_mask] = peak_ttp_map[valid_mask][:, 0]
    TTP_img = fill_nan_with_neighbors(TTP_img, peak_ttp_map, mask, neighborhood_size)
    TTP_img = denoise_ttp(TTP_img, mask, voxel_dims, z_thresholds, quantile_thresholds)
    return TTP_img


#########################################################
# data_io.py
def load_nifti(image_path):
    image_obj = nib.as_closest_canonical(nib.load(image_path))
    image_data = np.squeeze(image_obj.get_fdata(caching='unchanged'))
    affine_matrix = image_obj.affine
    return image_obj, image_data, affine_matrix
    
    
def load_image(subject_dir, modality, subject_id=None):
    """
    Load an image with flexible case handling for both the subject directory and modality.
    """
    if subject_id is None:
        subject_id = os.path.basename(os.path.normpath(subject_dir))
    
    # Try different case variations of the modality
    modalities_to_try = [modality, modality.lower(), modality.upper()]
    
    # 1. Check Preprocess Directory First (Standard pipeline location)
    preprocess_dir = os.path.join(subject_dir, 'preprocess')
    if os.path.exists(preprocess_dir):
        for mod in modalities_to_try:
            for ext in ['nii.gz', 'nii']:
                img_path = os.path.join(preprocess_dir, f'{subject_id}_{mod}.{ext}')
                if os.path.exists(img_path):
                    print(f"Found image in preprocess: {img_path}")
                    return load_nifti(img_path)

    # 2. Check Main Directory
    for mod in modalities_to_try:
        for ext in ['nii.gz', 'nii']:
            img_path = os.path.join(subject_dir, f'{subject_id}_{mod}.{ext}')
            if os.path.exists(img_path):
                print(f"Found image in root: {img_path}")
                return load_nifti(img_path)
    
    raise RuntimeError(f'No {subject_id}_{modality}.nii(.gz) found in {subject_dir} or {preprocess_dir}')

def load_json(subject_dir, modality, subject_id=None):
    if subject_id is None:
        subject_id = os.path.basename(os.path.normpath(subject_dir))

    mods = {modality, modality.lower(), modality.upper()}

    candidates = []
    for m in mods:
        candidates.extend([
            os.path.join(subject_dir, "preprocess", f"{subject_id}_{m}.json"),
            os.path.join(subject_dir, f"{subject_id}_{m}.json"),
            os.path.join(subject_dir, "preprocess", f"{m}.json"),
            os.path.join(subject_dir, f"{m}.json"),
        ])

    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)

    raise RuntimeError(f"Invalid or missing {modality} json for {subject_id}")


def create_new_nifti_image(image, reference, data_type=None):
    """
    Create a new NIfTI image using data from an existing NIfTI image as the reference for affine and header.

    Args:
    image (ndarray): The image data for the new NIfTI image.
    reference (Nifti1Image): The reference NIfTI image to copy affine and header information.
    data_type (data-type, optional): The data type for the new image. If None, the data type of image is used.

    Returns:
    Nifti1Image: A new NIfTI image.
    """
    if data_type is None:
        data_type = image.dtype  # Automatically detect data type from image if not provided

    reference.set_data_dtype(data_type)
    img_header = reference.header.copy()
    img_header['glmax'] = np.max(image)
    img_header['glmin'] = np.min(image)
    new_nifti_image = nib.Nifti1Image(image, reference.affine, img_header)

    return new_nifti_image

def save_image(image, reference, out_dir, modality, data_type=None, extension='nii.gz', subject_id=None):
    """
    Save a NIfTI image as <subject_id>_<modality>.<extension> under out_dir.
    If subject_id is None, fall back to the basename of out_dir (legacy behavior).
    """
    # Create new NIfTI
    nifti_image = create_new_nifti_image(image, reference, data_type)

    # Decide filename token
    if subject_id is None:
        subject_id = os.path.basename(os.path.normpath(out_dir))  # legacy fallback

    os.makedirs(out_dir, exist_ok=True)
    image_path = os.path.join(out_dir, f'{subject_id}_{modality}.{extension}')
    nib.save(nifti_image, image_path)
    return image_path

class ImageData:
    def __init__(self):
        self.data = {}

    def load_image(self, subject_dir, fname, subject_id=None):
        canon = fname.upper()  # normalize keys
        imgJ, img, affine = load_image(subject_dir, fname, subject_id=subject_id)
        self.data[canon + '_imgJ'] = imgJ
        self.data[canon + '_img']  = img
        self.data[canon + '_affine'] = affine

    def load_json(self, subject_dir, fname, subject_id=None):
        canon = fname.upper()
        self.data[canon + '_json'] = load_json(subject_dir, fname, subject_id=subject_id)

    # ⇩⇩⇩ put this back (it’s what your code is calling)
    def get_voxel_dims(self, key):
        """Return header zooms for the stored nib image object (e.g., 'PWI_imgJ')."""
        return tuple(self.data[key].header.get_zooms())

TIME_SLICE_REQUIRE = 20
def load_and_check_images(subject_dir, inputs, subject_id=None):
    images = ImageData()

    for fname in inputs:
        images.load_image(subject_dir, fname, subject_id=subject_id)
    try:
        images.load_json(subject_dir, 'pwi', subject_id=subject_id)
    except Exception:
        print('--- No PWI.json found, using defaults: TR=1.7, interleaved ascending slice order, EchoTime=0.03')
        images.data['PWI_json'] = {
            'RepetitionTime': 1.7,
            'EchoTime': 0.03,
        }

    if len(images.data['PWI_img'].shape) < 4:
        raise ValueError('PWI has dimension less than 4!')
    
    if images.data['PWI_img'].shape[3] < TIME_SLICE_REQUIRE:
        raise ValueError(f'PWI has less slices in time axis than {TIME_SLICE_REQUIRE}')

    if 'EchoTime' not in images.data['PWI_json']:
        print('--- No EchoTime in PWI.json, then set EchoTime 0.03 by default')
        images.data['PWI_json']['EchoTime'] = 0.03

    voxel_dims = tuple(images.get_voxel_dims('PWI_imgJ'))
    tr = float(images.data['PWI_json'].get('RepetitionTime', 1.7) or 1.7)
    if len(voxel_dims) < 4:
        voxel_dims = tuple(voxel_dims[:3]) + (tr,)
    elif not np.isfinite(voxel_dims[3]) or voxel_dims[3] <= 0:
        voxel_dims = tuple(voxel_dims[:3]) + (tr,)
    images.data['voxel_dims'] = voxel_dims
    # images.data['ADC_ss_img'] = images.data['BrainMask_img'] * images.data['ADC_img']

    return images    
#########################################################calculate_ttp_peaks#####################################
# Preprocessing function for PWI to TTP images
def preprocessing(subject_dir, images, config, output_dir=None, subject_id=None):
    """Process PWI to TTP images and register ADC to TTP and perform enantiomorphic transformation"""
    if subject_id is None:
        subject_id = Path(subject_dir).name  # "sub-02e8eb42"
    preprocess_dir = Path(output_dir) if output_dir else Path(subject_dir) / "preprocess"
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    
    # Update configuration with baseline settings
    config.update_baseline_config(images.data['PWI_img'].shape[-1])
    
    # Perform slice time correction
    images.data['PWI_STC_img'], _, _ = slicetime_correction(
        images.data['PWI_imgJ'], images.data['PWI_img'], images.data['PWI_json'], **config.slicetime_correction
    )
    images.data['PWI_STC_img_path'] = save_image(
        images.data['PWI_STC_img'],
        images.data['PWI_imgJ'],
        preprocess_dir,                # out_dir
        'PWI_STC',
        subject_id=subject_id
    )



    if config.motion_correction['motion_correction']:
        MC_affines_dir_path = preprocess_dir / f"{subject_id}_PWI_moco_aff"
        images.data['PWI_MC_img'] = handle_motion_correction(
            images.data['PWI_STC_img_path'], MC_affines_dir_path, config
        )
    else:
        images.data['PWI_MC_img'] = images.data['PWI_STC_img']
    images.data['PWI_MC_img'] = np.squeeze(images.data['PWI_MC_img'])
    if images.data['PWI_MC_img'].ndim == 3:
        images.data['PWI_MC_img'] = images.data['PWI_MC_img'][..., None]
    if images.data['PWI_MC_img'].ndim != 4:
        raise ValueError(f"Expected 4D PWI_MC_img, got {images.data['PWI_MC_img'].shape}")

    if config.motion_correction['save_corrected_PWI']:
        corrected_pwi_path = save_image(
            images.data['PWI_MC_img'],
            images.data['PWI_imgJ'],
            preprocess_dir,
            'PWI_moco',
            subject_id=subject_id
        )


    # Perform brain masking
    masking_method = config.brain_masking_method
    masking_config = config.brain_masking_config[masking_method]
    print(f"Using {masking_method} brain masking method!")

    #pwi_avg_img_path = save_image(images.data['PWI_MC_img'].mean(axis=3), images.data['PWI_imgJ'], subject_dir, 'PWI_average')
    pwi_avg = np.squeeze(images.data['PWI_MC_img'].mean(axis=-1))
    pwi_avg_img_path = save_image(
        pwi_avg,
        images.data['PWI_imgJ'],
        preprocess_dir,
        'PWI_average',
        subject_id=subject_id
    )
    if masking_method == 'synthstrip':
        #mask_file_path = masking_with_synthstrip(subject_dir, 'PWI_average', 'PWI_ss', 'PWI_mask', **masking_config)
        mask_file_path = preprocess_dir / f"{subject_id}_PWIbrain-mask.nii.gz"
        mask_ = generate_brain_mask_with_synthstrip(
            str(pwi_avg_img_path),
            str(mask_file_path),
            use_gpu=True,
            no_csf=False,
            model_path=None
        )
        
        _, images.data['PWI_mask_img'], _ = load_nifti(str(mask_file_path))
        images.data['PWI_mask_img'] = np.squeeze(images.data['PWI_mask_img'])
        pwi_avg_img_path = save_image(
            pwi_avg * images.data['PWI_mask_img'],
            images.data['PWI_imgJ'],
            preprocess_dir,
            'PWI_average',
            subject_id=subject_id
        )
    # Compute Gadolinium concentration map
    images.data['Gd_img'], images.data['S0'] = compute_gd_map(
        images.data['PWI_MC_img'], images.data['PWI_mask_img'], images.data['voxel_dims'], config.gd_concentration
    )

    # Compute peak time-to-peak map and TTP image
    images.data['peak_ttp_map'] = compute_peak_ttp_map_parallel(
        images.data['Gd_img'], images.data['PWI_mask_img'], images.data['voxel_dims'], config.ttp_peaks
    )
    images.data['TTP_img'] = compute_ttp(images.data['peak_ttp_map'], images.data['PWI_mask_img'], images.data['voxel_dims'][0:3])

    ## save TTP image rename it as TTP
    ttp_img_path = save_image(
        images.data['TTP_img'],
        images.data['PWI_imgJ'],
        preprocess_dir,
        'TTP',
        data_type=np.float32,
        subject_id=subject_id
    )

    print(f"TTP image of {subject_id} saved to {ttp_img_path}")


#########
def process_subject_sequential(subject_id, SubjDir, input_images, config, output_dir=None):
    try:
        print(f"Processing {subject_id}...")
        #subject_path = os.path.join(SubjDir, subject_id)
        
        # Load and validate images
        images = load_and_check_images(SubjDir, input_images, subject_id=subject_id)
        
        # Process
        preprocessing(SubjDir, images, config, output_dir=output_dir, subject_id=subject_id)
        
        # Clear memory after each subject
        del images
        import gc
        gc.collect()
        
        # Clear GPU memory if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
            
    except Exception as e:
        print(f"Error processing {subject_id}: {e}")
        return f"Error: {subject_id}: {e}"

###########################################################################
# # Sequential processing example
# if __name__ == "__main__":
#     input_images = ['PWI']
#     config = PWIPreprocessingConfig()
#     SubjDir = "/home/joshua/projects/ads_using/new/pwi_test/dataset/PWI_aug2024_resaved"
#     id_list = [d for d in os.listdir(SubjDir) if os.path.isdir(os.path.join(SubjDir, d)) and not d.startswith('.')]

#     for subject_id in id_list:
#         process_subject_sequential(subject_id, SubjDir, input_images, config)
