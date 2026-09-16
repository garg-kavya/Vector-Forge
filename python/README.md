# vectorforge (Python)

Python bindings for [VectorForge](../README.md): exact and HNSW k-nearest-neighbour search over
float32 vectors, with the C++20 core doing the work.

```python
import numpy as np
import vectorforge as vf

index = vf.Index(dim=128, metric="cosine", M=16, ef_construction=200)
index.add(np.random.default_rng(0).standard_normal((10_000, 128), dtype=np.float32))
labels, distances = index.search(np.ones(128, dtype=np.float32), k=5)
```

Install from a repository checkout (the build compiles the core in `..`):

```bash
pip install ./python
```

Reference: [docs/python-api.md](../docs/python-api.md).
