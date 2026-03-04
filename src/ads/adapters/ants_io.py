"""ANTs I/O and registration operations."""
from typing import Dict, Any
import ants
import numpy as np


class ANTsImage:
    """Wrapper for ANTs image operations."""

    # TODO: Fix PWI convert to MNI problem - 4D RAS reorientation commented out for now
    # @staticmethod
    # def _reorient_4d_to_ras(img_4d: ants.ANTsImage) -> ants.ANTsImage:
    #     """Reorient 4D image to RAS by processing each timeframe."""
    #     data_4d = img_4d.numpy()
    #     n_timepoints = data_4d.shape[-1]
    #
    #     # Extract spatial properties (3D only)
    #     origin_3d = tuple(list(img_4d.origin)[:3])
    #     spacing_3d = tuple(list(img_4d.spacing)[:3])
    #     direction_3d = img_4d.direction[:3, :3]  # 3x3 spatial direction matrix
    #
    #     # Extract temporal properties
    #     origin_temporal = img_4d.origin[-1] if len(img_4d.origin) > 3 else 0.0
    #     spacing_temporal = img_4d.spacing[-1] if len(img_4d.spacing) > 3 else 1.0
    #
    #     reoriented_frames = []
    #     for t in range(n_timepoints):
    #         # Extract 3D frame
    #         frame_3d = data_4d[..., t]
    #         frame_img = ants.from_numpy(
    #             frame_3d,
    #             origin=origin_3d,
    #             spacing=spacing_3d,
    #             direction=direction_3d  # Use 3x3 direction matrix
    #         )
    #
    #         # Reorient to RAS
    #         frame_ras = ants.reorient_image2(frame_img, orientation="RAS")
    #         reoriented_frames.append(frame_ras.numpy())
    #
    #     # Stack back to 4D
    #     data_4d_ras = np.stack(reoriented_frames, axis=-1)
    #
    #     # Prepare 4D properties (spatial + temporal)
    #     origin_4d = tuple(list(frame_ras.origin) + [origin_temporal])
    #     spacing_4d = list(frame_ras.spacing) + [spacing_temporal]
    #
    #     # Expand 3x3 direction to 4x4
    #     direction_4d = np.eye(4)
    #     direction_4d[:3, :3] = frame_ras.direction
    #
    #     # Create 4D image with reoriented properties
    #     return ants.from_numpy(
    #         data_4d_ras,
    #         origin=origin_4d,
    #         spacing=spacing_4d,
    #         direction=direction_4d,
    #         has_components=False
    #     )

    @staticmethod
    def load(path: str) -> ants.ANTsImage:
        """Load image and reorient to RAS."""
        img = ants.image_read(str(path))
        if len(img.shape) == 3:
            return ants.reorient_image2(img, orientation="RAS")
        # TODO: Fix PWI convert to MNI problem - 4D RAS reorientation commented out for now
        # elif len(img.shape) == 4:
        #     return ANTsImage._reorient_4d_to_ras(img)
        else:
            return img

    @staticmethod
    def save(img: ants.ANTsImage, path: str) -> None:
        """Save image with RAS orientation."""
        if len(img.shape) == 3:
            ants.reorient_image2(img, orientation="RAS").image_write(str(path))
        # TODO: Fix PWI convert to MNI problem - 4D RAS reorientation commented out for now
        # elif len(img.shape) == 4:
        #     ANTsImage._reorient_4d_to_ras(img).image_write(str(path))
        else:
            img.image_write(str(path))

    @staticmethod
    def binarize(img: ants.ANTsImage, threshold: float = 0.5) -> ants.ANTsImage:
        """Binarize image at threshold."""
        arr = (img.numpy() > threshold).astype("uint8")
        return ants.from_numpy(
            arr,
            origin=img.origin,
            spacing=img.spacing,
            direction=img.direction
        )

    @staticmethod
    def from_numpy(data, origin, spacing, direction):
        """Create ANTs image from numpy array."""
        return ants.from_numpy(
            data,
            origin=origin,
            spacing=spacing,
            direction=direction
        )


class ANTsRegistration:
    """ANTs registration operations."""

    @staticmethod
    def affine(
        fixed: ants.ANTsImage,
        moving: ants.ANTsImage,
        type_of_transform: str = 'Affine',
        reg_iterations: list = None,
        shrink_factors: list = None,
        smoothing_sigmas: list = None,
        grad_step: float = 0.1,
        metric: str = 'mattes',
        sampling: int = 32,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Perform affine registration."""
        return ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform=type_of_transform,
            reg_iterations=reg_iterations or [100, 50, 25],
            shrink_factors=shrink_factors or [8, 4, 2],
            smoothing_sigmas=smoothing_sigmas or [3.0, 2.0, 1.0],
            grad_step=grad_step,
            aff_metric=metric,
            aff_sampling=sampling,
            verbose=verbose,
        )

    @staticmethod
    def syn(
        fixed: ants.ANTsImage,
        moving: ants.ANTsImage,
        type_of_transform: str = 'SyN',
        reg_iterations: list = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Perform SyN registration."""
        return ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform=type_of_transform,
            reg_iterations=reg_iterations or [70, 50, 20],
            verbose=verbose,
        )

    @staticmethod
    def apply_transforms(
        fixed: ants.ANTsImage,
        moving: ants.ANTsImage,
        transforms: list,
        interpolator: str = 'linear'
    ) -> ants.ANTsImage:
        """Apply transforms to image."""
        return ants.apply_transforms(
            fixed=fixed,
            moving=moving,
            transformlist=transforms,
            interpolator=interpolator
        )


class ANTsTransformUtils:
    """Utilities for ANTs transform files."""

    @staticmethod
    def split_warp_and_affine(transform_list: list) -> tuple:
        """Split transform list into warp and affine components."""
        warp = None
        affine = None
        for t in transform_list:
            t_str = str(t)
            if t_str.endswith('.nii') or t_str.endswith('.nii.gz'):
                warp = t_str
            elif t_str.endswith('.mat'):
                affine = t_str
        if warp is None or affine is None:
            raise RuntimeError(f"Could not split transforms: {transform_list}")
        return warp, affine
