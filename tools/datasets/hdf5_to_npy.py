#!/usr/bin/env python3
"""Converts an ann-benchmarks HDF5 file into .npy files that `vectorforge` reads.

    python tools/datasets/hdf5_to_npy.py glove-100-angular.hdf5 out/glove

Writes <prefix>.base.npy (float32), <prefix>.queries.npy (float32), and, from the file's
neighbours/distances, <prefix>.gt.ids.npy (int64) and <prefix>.gt.distances.npy (float32) in the
layout `vectorforge search --gt <prefix>.gt` expects. The distances are converted to VectorForge's
definitions: angular files store cosine distance (kept), euclidean files store L2 distance
(squared here). Needs h5py and numpy.
"""

import pathlib
import sys

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover
    sys.exit("hdf5_to_npy.py needs h5py (pip install h5py)")


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    source = pathlib.Path(sys.argv[1])
    prefix = pathlib.Path(sys.argv[2])
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source, "r") as f:
        metric = f.attrs.get("distance", "unknown")
        base = np.ascontiguousarray(f["train"][:], dtype=np.float32)
        queries = np.ascontiguousarray(f["test"][:], dtype=np.float32)
        ids = np.asarray(f["neighbors"][:], dtype=np.int64)
        dist = np.asarray(f["distances"][:], dtype=np.float32)
    if metric == "euclidean":
        dist = dist * dist
    np.save(f"{prefix}.base.npy", base)
    np.save(f"{prefix}.queries.npy", queries)
    np.save(f"{prefix}.gt.ids.npy", ids)
    np.save(f"{prefix}.gt.distances.npy", dist)
    print(f"{source.name}: metric {metric}, base {base.shape}, queries {queries.shape}, "
          f"ground truth k = {ids.shape[1]} -> {prefix}.*.npy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
