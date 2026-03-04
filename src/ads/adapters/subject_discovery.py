"""Subject directory layout resolution.

Handles the conventions for finding input files within subject directories.
This adapter encapsulates knowledge of directory structure and naming patterns.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class InputOverrides:
    """Explicit paths that override auto-discovery."""
    dwi: Optional[Path] = None
    adc: Optional[Path] = None
    b0: Optional[Path] = None
    stroke: Optional[Path] = None
    pwi: Optional[Path] = None


class SubjectLayoutResolver:
    """Resolves file locations within subject directories.

    This class encapsulates the logic for finding input files based on:
    - Naming conventions (subject_id patterns)
    - Directory structure (root, modality subfolders)
    - File extensions (.nii.gz, .nii)
    - Search strategies (explicit → root → subfolder → recursive)
    """

    def __init__(
        self,
        subject_dir: Path,
        subject_id: str,
        base_id: str,
        overrides: Optional[InputOverrides] = None
    ):
        """Initialize layout resolver.

        Args:
            subject_dir: Path to subject directory
            subject_id: Full subject ID (e.g., "sub-02e8eb42")
            base_id: Base ID without prefix (e.g., "02e8eb42")
            overrides: Explicit path overrides
        """
        self.subject_dir = Path(subject_dir)
        self.subject_id = subject_id
        self.base_id = base_id
        self.overrides = overrides or InputOverrides()

    def find_image(self, image_type: str) -> Optional[Path]:
        """Find an image file with flexible naming conventions.

        Search strategy:
        1. Check explicit overrides (CLI or YAML)
        2. Search in subject root with standard patterns
        3. Search in modality subfolders (DWI/, dwi/)
        4. Recursive search as fallback

        Args:
            image_type: Type of image to find (DWI, ADC, B0, STROKE, etc.)

        Returns:
            Path to found image, or None if not found

        Raises:
            FileNotFoundError: If explicit override path doesn't exist
        """
        image_type_u = image_type.upper()

        # Check overrides first
        override_path = self._check_overrides(image_type_u)
        if override_path is not None:
            return override_path

        # Generate naming patterns
        patterns = self._generate_patterns(image_type)

        # Tier 1: Subject root
        found = self._search_in_directory(self.subject_dir, patterns)
        if found:
            logger.debug(f"Found {image_type} in subject root: {found}")
            return found

        # Tier 2: Modality subfolders
        for subfolder in ["DWI", "dwi", "PWI", "pwi"]:
            subdir = self.subject_dir / subfolder
            if subdir.is_dir():
                found = self._search_in_directory(subdir, patterns)
                if found:
                    logger.debug(f"Found {image_type} in {subfolder}/: {found}")
                    return found

        # Tier 3: Recursive fallback
        found = self._recursive_search(patterns)
        if found:
            return found

        logger.debug(f"No {image_type} file found with patterns: {patterns}")
        return None

    def _check_overrides(self, image_type: str) -> Optional[Path]:
        """Check if image type has an explicit override.

        Args:
            image_type: Image type (uppercase)

        Returns:
            Override path if exists and valid, None otherwise

        Raises:
            FileNotFoundError: If override path is set but doesn't exist
        """
        override_map = {
            "DWI": self.overrides.dwi,
            "ADC": self.overrides.adc,
            "B0": self.overrides.b0,
            "STROKE": self.overrides.stroke,
            "STROKE_FIXED": self.overrides.stroke,
        }

        override_path = override_map.get(image_type)
        if override_path is None:
            return None

        # Resolve relative paths
        if not override_path.is_absolute():
            override_path = self.subject_dir / override_path

        if not override_path.exists():
            raise FileNotFoundError(
                f"{image_type} path set explicitly but does not exist: {override_path}"
            )

        logger.info(f"Using explicit {image_type} path: {override_path}")
        return override_path

    def _generate_patterns(self, image_type: str) -> list[str]:
        """Generate filename patterns for an image type.

        Args:
            image_type: Image type (DWI, ADC, B0, etc.)

        Returns:
            List of filename patterns to try
        """
        patterns = []
        type_variants = []
        for candidate in [image_type, image_type.upper(), image_type.lower()]:
            if candidate not in type_variants:
                type_variants.append(candidate)

        # Subject ID patterns
        for id_variant in [self.subject_id, self.base_id]:
            for type_variant in type_variants:
                for ext in [".nii.gz", ".nii"]:
                    patterns.append(f"{id_variant}_{type_variant}{ext}")

        # Fallback plain names without subject prefix (e.g., ADC.nii.gz, DWI.nii.gz)
        for type_variant in type_variants:
            for ext in [".nii.gz", ".nii"]:
                patterns.append(f"{type_variant}{ext}")

        return patterns

    def _search_in_directory(
        self,
        directory: Path,
        patterns: list[str]
    ) -> Optional[Path]:
        """Search for patterns in a single directory.

        Args:
            directory: Directory to search
            patterns: List of filename patterns

        Returns:
            First matching path, or None
        """
        for pattern in patterns:
            path = directory / pattern
            if path.exists():
                return path
        return None

    def _recursive_search(self, patterns: list[str]) -> Optional[Path]:
        """Recursively search for patterns under subject directory.

        Args:
            patterns: List of filename patterns

        Returns:
            Best matching path (shortest relative path, prefer .nii.gz), or None
        """
        hits = []
        for pattern in patterns:
            hits.extend(self.subject_dir.rglob(pattern))

        if not hits:
            return None

        # Deterministic selection: shortest relative path, prefer .nii.gz
        hits.sort(
            key=lambda p: (
                len(p.relative_to(self.subject_dir).parts),
                0 if p.name.endswith(".nii.gz") else 1,
                str(p),
            )
        )

        chosen = hits[0]
        if len(hits) > 1:
            logger.warning(
                f"Multiple candidates found. Using: {chosen}. "
                f"Candidates: {[str(p) for p in hits]}"
            )

        return chosen
