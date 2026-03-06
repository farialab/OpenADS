"""OpenADS package metadata and bootstrap helpers."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"

# Ensure `ads` modules are importable when the package is imported directly.
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

__version__ = "1.0.0"
__author__ = "Joshua Shun Liu, Chin-Fu Liu, Andreia V. Faria"

__all__ = [
    "__version__",
    "__author__",
    "PROJECT_ROOT",
    "SRC_ROOT",
]
