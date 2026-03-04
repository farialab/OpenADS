"""Adapters layer - handles I/O and external library integrations.

This layer provides clean interfaces to external systems and libraries,
isolating the rest of the codebase from implementation details.
"""

from .subject_discovery import SubjectLayoutResolver, InputOverrides

__all__ = ["SubjectLayoutResolver", "InputOverrides"]
