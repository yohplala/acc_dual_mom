"""Cross-module exception types for the fetch / stitching pipeline.

Lives in its own module so both `fetch.py` and `stitching.py` can import
`FetchError` without a circular dependency between them.
"""

from __future__ import annotations


class FetchError(RuntimeError):
    """Raised when an upstream price provider returns unusable data, when
    a stitch precondition fails, or when an FX conversion can't proceed."""
