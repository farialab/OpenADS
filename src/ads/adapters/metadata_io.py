"""JSON sidecar file handling for PWI data.

Handles discovery and copying of JSON metadata files that accompany NIfTI images.
"""

from __future__ import annotations
import logging
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class JsonSidecarHandler:
    """Handles JSON sidecar file discovery and operations."""

    @staticmethod
    def find_json(
        subject_dir: Path,
        subject_id: str,
        base_id: str,
        modality: str
    ) -> Optional[Path]:
        """Find JSON sidecar file for a modality.

        Args:
            subject_dir: Subject directory to search
            subject_id: Subject ID (e.g., 'sub-02e8eb42')
            base_id: Base ID without 'sub-' prefix
            modality: Modality name (e.g., 'PWI')

        Returns:
            Path to JSON file if found, None otherwise
        """
        if not subject_dir.exists() or not subject_dir.is_dir():
            return None

        modality_lower = modality.lower()
        subject_matches: list[Path] = []
        modality_matches: list[Path] = []

        for filepath in subject_dir.iterdir():
            if not filepath.is_file() or not filepath.name.endswith(".json"):
                continue

            fname_lower = filepath.name.lower()

            # Check if filename contains subject ID and modality
            if ((base_id.lower() in fname_lower or subject_id.lower() in fname_lower) and
                modality_lower in fname_lower):
                subject_matches.append(filepath)
                continue

            # Fallback: modality-only naming (e.g., PWI.json)
            stem_lower = filepath.stem.lower()
            if modality_lower in fname_lower or stem_lower == modality_lower:
                modality_matches.append(filepath)

        if subject_matches:
            subject_matches.sort(key=lambda p: (0 if p.name.endswith(".json") else 1, str(p)))
            chosen = subject_matches[0]
            if len(subject_matches) > 1:
                logger.warning(
                    "Multiple subject-matched JSON sidecars found. Using %s. Candidates: %s",
                    chosen.name,
                    [p.name for p in subject_matches],
                )
            logger.info(f"Found JSON sidecar: {chosen.name}")
            return chosen

        if modality_matches:
            modality_matches.sort(key=lambda p: (len(p.name), str(p)))
            chosen = modality_matches[0]
            if len(modality_matches) > 1:
                logger.warning(
                    "No subject-matched JSON found; multiple modality-only JSON sidecars found. "
                    "Using %s. Candidates: %s",
                    chosen.name,
                    [p.name for p in modality_matches],
                )
            else:
                logger.info("Found modality-only JSON sidecar: %s", chosen.name)
            return chosen

        return None

    @staticmethod
    def copy_json(source: Path, dest: Path) -> Path:
        """Copy JSON file to destination.

        Args:
            source: Source JSON file path
            dest: Destination file path

        Returns:
            Path to copied file

        Raises:
            FileNotFoundError: If source file doesn't exist
        """
        if not source.exists():
            raise FileNotFoundError(f"Source JSON file not found: {source}")

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        logger.info(f"Copied JSON sidecar: {source.name} -> {dest.name}")

        return dest
