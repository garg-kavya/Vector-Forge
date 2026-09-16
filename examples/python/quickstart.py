"""VectorForge Python quickstart: build, search, remove, save and reload an index.

    pip install ./python
    python examples/python/quickstart.py
"""

import pathlib
import tempfile
import time

import numpy as np

import vectorforge as vf


def main() -> None:
    rng = np.random.default_rng(0)
    dim = 64
    xb = rng.standard_normal((20_000, dim), dtype=np.float32)
    xq = rng.standard_normal((100, dim), dtype=np.float32)

    index = vf.Index(dim=dim, metric="cosine", index="hnsw", M=16, ef_construction=200, seed=42)
    start = time.perf_counter()
    index.add(xb, np.arange(len(xb), dtype=np.uint64), num_threads=0)  # all cores
    print(f"built {len(index)} vectors in {time.perf_counter() - start:.2f} s "
          f"(SIMD tier: {vf.simd_level()})")

    labels, distances = index.search(xq, k=10, ef_search=128)
    print("first query:", labels[0][:5], distances[0][:5])

    # Exact results from a Flat index, for a recall estimate.
    exact = vf.Index(dim=dim, metric="cosine", index="flat")
    exact.add(xb)
    truth, _ = exact.search(xq, k=10)
    recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(labels, truth)])
    print(f"recall@10 = {recall:.3f}")

    index.remove([int(labels[0][0])])
    print("removed", int(labels[0][0]), "->", int(labels[0][0]) in index)

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "quickstart.vfidx"
        index.save(path)
        loaded = vf.Index.load(path, mmap=True)
        again, _ = loaded.search(xq[:1], k=10, ef_search=128)
        print("reloaded:", len(loaded), "vectors; same results:",
              bool((again[0] == index.search(xq[:1], k=10, ef_search=128)[0][0]).all()))
        print("stats:", {k: v for k, v in loaded.stats().items() if k != "memory"})
        del loaded  # release the file mapping before the directory is removed


if __name__ == "__main__":
    main()
