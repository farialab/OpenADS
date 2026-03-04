"""Registration service - pure business logic."""
from typing import Dict, Optional
import ants
from ads.domain.registration_spec import AffineSpec, SyNSpec
from ads.adapters.ants_io import ANTsRegistration, ANTsImage


class RegistrationService:
    """Handles registration computations (no I/O)."""

    @staticmethod
    def perform_affine(
        fixed: ants.ANTsImage,
        moving: ants.ANTsImage,
        spec: AffineSpec
    ) -> Dict:
        """Perform affine registration with given spec."""
        return ANTsRegistration.affine(
            fixed=fixed,
            moving=moving,
            type_of_transform=spec.type_of_transform,
            reg_iterations=spec.reg_iterations,
            shrink_factors=spec.shrink_factors,
            smoothing_sigmas=spec.smoothing_sigmas,
            grad_step=spec.grad_step,
            metric=spec.metric,
            sampling=spec.sampling,
            verbose=spec.verbose,
        )

    @staticmethod
    def perform_syn(
        fixed: ants.ANTsImage,
        moving: ants.ANTsImage,
        spec: SyNSpec
    ) -> Dict:
        """Perform SyN registration with given spec."""
        return ANTsRegistration.syn(
            fixed=fixed,
            moving=moving,
            type_of_transform=spec.type_of_transform,
            reg_iterations=spec.reg_iterations,
            verbose=spec.verbose,
        )

    @staticmethod
    def apply_affine_to_images(
        fixed: ants.ANTsImage,
        images: Dict[str, Optional[ants.ANTsImage]],
        transform: list,
    ) -> Dict[str, Optional[ants.ANTsImage]]:
        """Apply affine transform to multiple images."""
        results = {}
        for key, img in images.items():
            if img is None:
                results[key] = None
                continue

            interpolator = 'nearestNeighbor' if key in ['mask', 'stroke'] else 'linear'
            registered = ANTsRegistration.apply_transforms(
                fixed=fixed,
                moving=img,
                transforms=transform,
                interpolator=interpolator
            )

            # Binarize masks
            if key in ['mask', 'stroke']:
                registered = ANTsImage.binarize(registered)

            results[key] = registered

        return results

    @staticmethod
    def apply_syn_to_images(
        fixed: ants.ANTsImage,
        images: Dict[str, Optional[ants.ANTsImage]],
        transforms: list,
    ) -> Dict[str, Optional[ants.ANTsImage]]:
        """Apply SyN transforms to multiple images."""
        results = {}
        for key, img in images.items():
            if img is None:
                results[key] = None
                continue

            interpolator = 'nearestNeighbor' if key in ['mask', 'stroke'] else 'linear'
            registered = ANTsRegistration.apply_transforms(
                fixed=fixed,
                moving=img,
                transforms=transforms,
                interpolator=interpolator
            )

            # Binarize masks
            if key in ['mask', 'stroke']:
                registered = ANTsImage.binarize(registered)

            results[key] = registered

        return results
