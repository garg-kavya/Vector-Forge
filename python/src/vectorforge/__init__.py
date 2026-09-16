"""VectorForge: exact and HNSW vector similarity search with a C++20 core.

    >>> import numpy as np, vectorforge as vf
    >>> index = vf.Index(dim=4, metric="cosine")
    >>> index.add(np.eye(4, dtype=np.float32))
    4
    >>> labels, distances = index.search(np.eye(4, dtype=np.float32)[0], k=2)

See docs/python-api.md for the full reference.
"""

from ._vectorforge import (
    INVALID_ID,
    CorruptIndexError,
    DuplicateIdError,
    Index,
    VectorForgeError,
    __version__,
    git_sha,
    simd_level,
)

__all__ = [
    "INVALID_ID",
    "CorruptIndexError",
    "DuplicateIdError",
    "Index",
    "VectorForgeError",
    "__version__",
    "git_sha",
    "simd_level",
]
