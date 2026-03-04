"""Memory utility helpers."""

from __future__ import annotations

from typing import Any

try:
    from torch import Tensor  # type: ignore
except Exception:  # pragma: no cover
    Tensor = None  # type: ignore

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


def num_bytes(data: Any) -> int:
    r"""Return memory bytes consumed by tensor-like/container data."""
    if data is None:
        return 0

    if Tensor is not None and isinstance(data, Tensor):
        return int(data.element_size() * data.numel())

    if np is not None and isinstance(data, np.ndarray):
        return int(data.nbytes)

    if isinstance(data, (bytes, bytearray, memoryview)):
        return int(len(data))

    if isinstance(data, str):
        return int(len(data.encode("utf-8")))

    if isinstance(data, (list, tuple, set)):
        return sum(num_bytes(value) for value in data)

    if isinstance(data, dict):
        return sum(num_bytes(value) for value in data.values())

    raise NotImplementedError(f"'num_bytes' not implemented for '{type(data)}'")
