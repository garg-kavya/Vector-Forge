# Python API

`import vectorforge as vf` exposes one class, `vf.Index`, over the C++ `vf::Collection`. Design:
[DESIGN.md §14](DESIGN.md#14-python-api-design). Source: `python/src/vectorforge/`
(pybind11 3.1.0, scikit-build-core 1.0.3). Runnable example:
[examples/python/quickstart.py](../examples/python/quickstart.py).

## Installing

From a repository checkout (the build compiles the C++ core in the repository root):

```bash
pip install ./python            # needs a C++20 compiler and CMake >= 3.25
pip install "./python[test]" && pytest python/tests
```

On Windows, build from a Visual Studio developer prompt (MSVC matches the python.org CPython ABI;
MinGW is not supported for the extension). Tested on the development machine with CPython 3.12
(Linux, GCC) and 3.14 (Windows, MSVC); `.github/workflows/python.yml` runs both versions on both
systems. An sdist of `python/` alone does not contain the core; build wheels from a checkout.

For development, `cmake -DVF_BUILD_PYTHON=ON` builds the module in-tree (pybind11 is fetched,
pinned by SHA-256) and adds a `python.pytest` CTest test.

## Reference

```python
vf.Index(*, dim, metric="l2", index="hnsw", M=16, ef_construction=200, ef_search=50,
         seed=0x5EEDF0A6E, normalize=False, concurrency="concurrent")
```

`metric`: `"l2"` (squared distance), `"ip"` / `"inner_product"` (distance `−⟨a, b⟩`), `"cosine"`
(`1 − cos`, vectors normalised on insert and query). `index`: `"hnsw"` or `"flat"` (exact).
`concurrency`: `"concurrent"` or `"coarse"` (docs/concurrency.md). All arguments are keyword-only.

| Member | Behaviour |
|---|---|
| `add(vectors, ids=None, *, upsert=False, num_threads=0, strict=False) -> int` | Inserts `(n, d)` or `(d,)` vectors. `ids`: any integer array-like of length n (non-negative, below 2⁶⁴ − 1); omitted ids are row numbers continuing after every row stored so far. The batch is validated as a whole first. Returns the count. |
| `search(queries, *, k=10, ef_search=None, num_threads=0, strict=False) -> (labels, distances)` | `(n, d)` queries give `uint64 (n, k)` labels and `float32 (n, k)` distances, ascending; a `(d,)` query gives `(k,)` arrays. Missing results are padded with `vf.INVALID_ID` (2⁶⁴ − 1) and `inf`. |
| `remove(ids)` | Removes ids one by one (a scalar or an array-like); raises `KeyError` at the first unknown id, after removing the ones before it. |
| `get(id) -> ndarray` | Copy of the stored vector (normalised for cosine/`normalize=True`). |
| `id in index`, `len(index)` | Membership and live count. |
| `save(path)`, `Index.load(path, *, mmap=True, verify="auto", concurrency="concurrent")` | `.vfidx` files ([storage-format.md](storage-format.md)); `verify`: `"auto"`, `"none"`, `"metadata"`, `"full"`. |
| `compact() -> dict` | Rebuilds without removed vectors. |
| `stats() -> dict`, `config` | Counts, memory breakdown, SIMD tier; the configuration. |
| `dim`, `metric`, `index_type` | Read-only properties. |
| `vf.simd_level()` | `"avx2"` or `"scalar"`; raises if `VF_SIMD` asks for something unavailable. |
| `vf.__version__`, `vf.git_sha` | Build identification. |

`num_threads`: `0` uses every hardware thread, `1` runs on the calling thread; pools are created
once per thread count and shared. `strict=True` raises `TypeError` unless the input already is a
C-contiguous float32 array.

## Copies, the GIL and threads

- A float32 C-contiguous input is read in place; any other array-like is converted once by NumPy
  (a float64 array costs one converted copy). `add` then copies each row into the index once.
  `search` allocates the two result arrays and the core writes into them directly.
- `add`, `search`, `save`, `load` and `compact` release the GIL while the C++ code runs; the
  argument arrays stay referenced for the whole call. Do not modify an input array from another
  thread during a call.
- An `Index` may be used from several Python threads at once (the collection is thread-safe:
  searches run in parallel with each other and with inserts, docs/concurrency.md).
- Arrays returned by `get` and `search` are copies owned by NumPy; nothing returned by the module
  points into a memory-mapped file, so closing an index (dropping the last reference) is always
  safe. On Windows a mapped `.vfidx` file cannot be deleted or replaced while an `Index` loaded
  from it is alive.

## Errors

| Situation | Exception |
|---|---|
| invalid argument, dimension mismatch, non-finite value, zero vector with cosine | `ValueError` |
| unknown id (`get`, `remove`) | `KeyError` |
| id already present without `upsert` | `vf.DuplicateIdError` (subclass of `vf.VectorForgeError` and `ValueError`) |
| file cannot be opened or written | `OSError` |
| damaged file or unsupported format version | `vf.CorruptIndexError` (subclass of `vf.VectorForgeError`) |
| wrong argument types, non-integer ids, `strict=True` violations | `TypeError` |
| collection capacity exhausted, out of memory | `MemoryError` |

A failed `add` of a batch inserts nothing if validation fails; see `Collection::add_batch` for
failures after validation.

## Overhead

Ryzen 7 4800H, CPython 3.14, HNSW 100 000 × 128-d, k = 10, ef_search = 64, one thread
([results](../benchmarks/results/2026-09-17_ryzen7-4800h_msvc-release_phase8/README.md)): one batch
`search` call runs at 9 717 queries/s against 9 657 for the same call in C++; a Python loop of
single-query calls runs at 9 155 queries/s, i.e. 6.3 µs of overhead per call.
