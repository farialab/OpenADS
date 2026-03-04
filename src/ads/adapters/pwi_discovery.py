"""PWI-specific subject layout resolution.

Extends SubjectLayoutResolver with PWI-specific patterns including:
- HP_manual2 → HP_manual aliasing
- PWI-specific file naming conventions
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Dict

from .subject_discovery import SubjectLayoutResolver

logger = logging.getLogger(__name__)


class PWILayoutResolver(SubjectLayoutResolver):
    """Subject layout resolver with PWI-specific patterns.

    Handles PWI-specific naming conventions and aliases that differ
    from standard DWI patterns.
    """

    # Mapping of internal names to output filenames
    MODALITY_ALIASES: Dict[str, str] = {
        "HP_manual2": "HP_manual",  # Rename HP_manual2 to HP_manual
        "PWI": "PWI",
    }

    def _norm_mod(self, modality: str) -> str:
        return (modality or "").strip().upper()

    def find_image_with_alias(self, image_type: str) -> Optional[Path]:
        """Find image with alias handling.

        First tries the requested image_type, then falls back to checking
        aliases. For example, if "HP_manual" is requested, also checks for
        "HP_manual2".

        Args:
            image_type: Image type to find (e.g., "HP_manual")

        Returns:
            Path to image file if found, None otherwise
        """

        # Try direct match first
        path = self.find_image(image_type)
        if path:
            return path

        # Try reverse alias lookup
        # If looking for "HP_manual", also try "HP_manual2"
        for alias, target in self.MODALITY_ALIASES.items():
            if target.lower() == image_type.lower():
                alt_path = self.find_image(alias)
                if alt_path:
                    logger.info(f"Found {image_type} via alias {alias}: {alt_path.name}")
                    return alt_path

        return None

    def get_output_name(self, image_type: str) -> str:
        """Get the standardized output name for an image type.

        Applies aliasing rules to ensure consistent output naming.

        Args:
            image_type: Input image type (may be alias)

        Returns:
            Standardized output name

        Example:
            >>> resolver.get_output_name("HP_manual2")
            "HP_manual"
        """
        return self.MODALITY_ALIASES.get(image_type, image_type)
