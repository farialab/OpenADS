"""Output cleanup helpers for subject-level pipeline artifacts.

This module performs hard deletion (unlink/rmtree), not trash moves.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, Iterable, List

import yaml


@dataclass
class CleanupStats:
    deleted_files: int = 0
    deleted_dirs: int = 0
    kept_files: int = 0


def _load_keep_manifest(path: Path) -> Dict[str, Dict[str, List[str]]]:
    if not path.exists():
        raise FileNotFoundError(f"keep_files manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    outputs = data.get("outputs", {})
    if not isinstance(outputs, dict):
        raise ValueError("Invalid keep_files manifest: 'outputs' must be a mapping.")
    return outputs


def _patterns_for(rel_parts: Iterable[str], keep_map: Dict[str, List[str]]) -> List[str]:
    parts = list(rel_parts)
    if not parts:
        return keep_map.get("*", [])
    stage = parts[0]
    patterns = []
    patterns.extend(keep_map.get("*", []))
    patterns.extend(keep_map.get(stage, []))
    return patterns


def _matches_keep(rel_path: Path, patterns: List[str]) -> bool:
    rel_s = rel_path.as_posix()
    name = rel_path.name
    for p in patterns:
        if fnmatch(rel_s, p) or fnmatch(name, p):
            return True
    return False


def cleanup_subject_outputs(
    *,
    output_root: Path,
    subject_id: str,
    modality: str,
    keep_manifest: Path,
    logger=None,
) -> CleanupStats:
    """Delete intermediate files not listed in keep manifest for one modality."""
    stats = CleanupStats()
    modality_u = modality.upper()
    modality_dir = output_root / subject_id / modality_u
    if not modality_dir.exists():
        return stats

    outputs = _load_keep_manifest(keep_manifest)
    keep_map = outputs.get(modality_u, {})
    if not isinstance(keep_map, dict):
        raise ValueError(f"Invalid keep section for modality {modality_u}")

    deleted_file_paths: List[Path] = []
    deleted_dir_paths: List[Path] = []

    # Delete non-keep files first.
    for f in sorted(modality_dir.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(modality_dir)
        patterns = _patterns_for(rel.parts, keep_map)
        if _matches_keep(rel, patterns):
            stats.kept_files += 1
            continue
        f.unlink(missing_ok=True)
        stats.deleted_files += 1
        deleted_file_paths.append(rel)

    # Then delete empty directories bottom-up.
    for d in sorted(modality_dir.rglob("*"), reverse=True):
        if d.is_dir():
            try:
                d.rmdir()
                stats.deleted_dirs += 1
                deleted_dir_paths.append(d.relative_to(modality_dir))
            except OSError:
                pass

    if logger is not None:
        logger.info(
            "[CLEANUP:%s] kept=%d deleted_files=%d deleted_dirs=%d",
            modality_u,
            stats.kept_files,
            stats.deleted_files,
            stats.deleted_dirs,
        )
        for rel_path in deleted_file_paths:
            logger.info("[CLEANUP:%s] deleted file: %s", modality_u, rel_path.as_posix())
        for rel_path in deleted_dir_paths:
            logger.info("[CLEANUP:%s] deleted dir: %s", modality_u, rel_path.as_posix())
    return stats
