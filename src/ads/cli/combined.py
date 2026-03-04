"""Combined CLI entrypoint."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence


def run(argv: Sequence[str]) -> int:
    """Run combined DWI+PWI workflow via existing script entrypoint."""
    project_root = Path(__file__).resolve().parents[3]
    script = project_root / "scripts" / "run_ads_combined.py"
    cmd = [sys.executable, str(script), *list(argv)]
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)
