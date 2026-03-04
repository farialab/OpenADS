"""PWI coregistration service - pure business logic."""
from typing import Dict, Optional
import ants
import numpy as np
from ads.domain.pwi_registration_spec import PWICoregSpec
from ads.adapters.ants_io import ANTsRegistration, ANTsImage


class PWICoregistrationService:
    """Handles PWI→DWI coregistration (no I/O)."""

    @staticmethod
    def register_pwi_to_dwi(
        pwi_mask: ants.ANTsImage,
        dwi_mask: ants.ANTsImage,
        spec: PWICoregSpec
    ) -> Dict:
        """Perform PWI→DWI affine registration."""
        return ANTsRegistration.affine(
            fixed=dwi_mask,
            moving=pwi_mask,
            type_of_transform=spec.type_of_transform,
            verbose=spec.verbose,
        )

    @staticmethod
    def apply_transforms_to_pwi(
        fixed: ants.ANTsImage,
        images: Dict[str, Optional[ants.ANTsImage]],
        transforms: list,
        interpolator_images: str = 'linear',
        interpolator_masks: str = 'nearestNeighbor',
    ) -> Dict[str, Optional[ants.ANTsImage]]:
        """Apply transforms to PWI images."""
        results = {}
        for key, img in images.items():
            if img is None:
                results[key] = None
                continue

            is_mask = 'HP' in key or 'mask' in key.lower()
            interpolator = interpolator_masks if is_mask else interpolator_images

            registered = ANTsRegistration.apply_transforms(
                fixed=fixed,
                moving=img,
                transforms=transforms,
                interpolator=interpolator
            )

            if is_mask:
                registered = ANTsImage.binarize(registered)

            results[key] = registered

        return results

    @staticmethod
    def apply_brain_mask_numpy(
        image_data: np.ndarray,
        mask_data: np.ndarray,
    ) -> np.ndarray:
        """Apply brain mask to image data (pure NumPy operation)."""
        if image_data.shape != mask_data.shape:
            raise ValueError(
                f"Shape mismatch: image {image_data.shape} vs mask {mask_data.shape}"
            )

        mask_binary = mask_data > 0.5
        out = np.zeros_like(image_data, dtype=np.float32)
        out[mask_binary] = image_data[mask_binary].astype(np.float32)
        return out

    @staticmethod
    def transform_4d_timeseries(
        fixed: ants.ANTsImage,
        moving_4d: ants.ANTsImage,
        transforms: list,
        interpolator: str = 'linear',
    ) -> ants.ANTsImage:
        """
        Transform 4D time series frame by frame.

        Args:
            fixed: Fixed image (3D)
            moving_4d: Moving 4D time series
            transforms: Transform list
            interpolator: Interpolation method

        Returns:
            Transformed 4D image
        """
        import numpy as np

        # Get 4D data
        data_4d = moving_4d.numpy()
        n_timepoints = data_4d.shape[-1]

        # Initialize output
        output_frames = []

        # Transform each frame
        for t in range(n_timepoints):
            # Extract 3D frame
            frame_3d = data_4d[..., t]
            frame_img = ants.from_numpy(
                frame_3d,
                origin=moving_4d.origin,
                spacing=moving_4d.spacing,
                direction=moving_4d.direction
            )

            # Apply transforms
            transformed = ants.apply_transforms(
                fixed=fixed,
                moving=frame_img,
                transformlist=transforms,
                interpolator=interpolator
            )

            output_frames.append(transformed.numpy())

        # Stack frames back to 4D
        output_4d = np.stack(output_frames, axis=-1)

        # Create 4D ANTs image
        return ants.from_numpy(
            output_4d,
            origin=fixed.origin,
            spacing=list(fixed.spacing) + [moving_4d.spacing[-1]],  # Keep time spacing
            direction=fixed.direction,
            has_components=False
        )
