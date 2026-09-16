# Phase 8 Python overhead — 2026-09-17

Raw data: [`python_overhead_hnsw_100k_d128.json`](python_overhead_hnsw_100k_d128.json)
(`benchmarks/python/bench_overhead.py`).

## Environment and protocol

Ryzen 7 4800H, Windows 11, MSVC 19.50 (both the `msvc-release` C++ build and the extension module
built by `pip install ./python`), AVX2 tier, CPython 3.14.3, NumPy 2.5.3. Source
`1b9cee572c9a-dirty`, the working tree that became the Phase 8 commit.

The same saved HNSW index (100 000 × 128-d Gaussian mixture, M = 16, ef_construction = 200, the
Phase 6a file) and the same 5 000 queries (`vectorforge gen-data --n 5000 --dim 128 --seed 2`) are
searched with k = 10, ef_search = 64, one thread:

- C++: `vf_bench --scenario threads --threads 1 --queries-file queries.npy` (one
  `search_batch` call per pass);
- Python batch: `index.search(queries, k=10, ef_search=64, num_threads=1)`;
- Python per query: a Python loop calling `index.search(q, ...)` for each 1-D query.

Each variant: warm-up, then 5 passes; the median pass is reported (min–max in the file).

## Results

| Variant | QPS (median) | range |
|---|---|---|
| C++ `search_batch` | 9 657 | — |
| Python, one batch call | 9 717 | 9 655–9 775 |
| Python, one call per query | 9 155 | 9 075–9 222 |

- A batch call from Python costs the same as the C++ call (+0.6%, within run-to-run variation):
  the float32 C-contiguous input is used in place (verified by `test_io.py`) and results are
  written straight into the returned NumPy arrays.
- Calling once per query adds 6.3 µs per call (argument conversion, two result arrays, GIL
  release/reacquire) — 6% of a 0.1 ms HNSW query here; batching removes it.

## Reproduce

```powershell
cmake --build --preset msvc-release --target vf_bench vectorforge_cli
pip install ./python
out\build\msvc-release\apps\cli\vectorforge.exe gen-data --n 5000 --dim 128 --seed 2 --out queries.npy
out\build\msvc-release\benchmarks\vf_bench.exe --scenario save --n 100000 --dim 128 --index hnsw --index-file hnsw_d128.vfidx
python benchmarks\python\bench_overhead.py --vf-bench out\build\msvc-release\benchmarks\vf_bench.exe --index hnsw_d128.vfidx --queries queries.npy --out python_overhead_hnsw_100k_d128.json
```
