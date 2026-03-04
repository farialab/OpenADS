"""Manifest generation service."""
import json
import yaml
import hashlib
import platform
import datetime as dt
from pathlib import Path
from typing import Dict, Any
from dataclasses import asdict
from ads.domain.registration_data import RegistrationResult, AffineResult, SyNResult


class ManifestService:
    """Generate registration manifest files."""

    @staticmethod
    def create_manifest(
        subject_id: str,
        result: RegistrationResult,
        inputs: Dict[str, Any],
        templates: Dict[str, Any],
        config: Dict[str, Any],
        ants_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create manifest dictionary."""
        now = dt.datetime.now().astimezone().isoformat()

        manifest = {
            'subject_id': subject_id,
            'timestamp': now,
            'inputs': inputs,
            'templates': templates,
            'outputs': ManifestService._serialize_outputs(result),
            'config': config,
            'ants': ants_results,
            'environment': {
                'python': platform.python_version(),
                'platform': platform.platform(),
                'antsPy': 'unknown',  # Could import ants.__version__ if needed
            },
            'hashes': ManifestService._compute_hashes(result),
        }

        return manifest

    @staticmethod
    def write_manifest(manifest: Dict[str, Any], yaml_path: Path, json_path: Path) -> None:
        """Write manifest to YAML and JSON files."""
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(manifest, f, sort_keys=False)

        with open(json_path, 'w') as f:
            json.dump(manifest, f, indent=2)

    @staticmethod
    def _serialize_outputs(result: RegistrationResult) -> Dict[str, Any]:
        """Convert RegistrationResult to serializable dict."""
        def path_to_str(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif obj is None:
                return None
            elif isinstance(obj, (AffineResult, SyNResult)):
                d = asdict(obj)
                return {k: str(v) if isinstance(v, Path) else v for k, v in d.items()}
            else:
                return obj

        return {
            'affine': path_to_str(result.affine),
            'syn': path_to_str(result.syn),
            'manifest_yaml': str(result.manifest_yaml),
            'manifest_json': str(result.manifest_json),
        }

    @staticmethod
    def _compute_hashes(result: RegistrationResult) -> Dict[str, str]:
        """Compute SHA256 hashes for key output files."""
        hashes = {}

        # Hash key files from final SyN output
        key_files = [
            result.syn.dwi,
            result.syn.adc,
            result.syn.mask,
            result.syn.affine_fwd,
            result.syn.warp_fwd,
        ]

        for file_path in key_files:
            if file_path is not None and Path(file_path).exists():
                hashes[str(file_path)] = ManifestService._sha256(Path(file_path))

        return hashes

    @staticmethod
    def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
        """Compute SHA256 hash of file."""
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            while chunk := f.read(chunk_size):
                h.update(chunk)
        return h.hexdigest()
