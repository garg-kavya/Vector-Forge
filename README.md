# VectorForge

VectorForge is a C++20 vector similarity search engine built from first principles: exact
(brute-force) and approximate (HNSW) k-nearest-neighbour search over float32 vectors, with SIMD
distance kernels, a thread pool, a versioned on-disk format with memory-mapped vectors, an HTTP API
and Python bindings.

> **Status:** early development — Phases 0–8 of the [engineering design](docs/DESIGN.md) are
> complete: toolchain; core types, deterministic RNG, scalar distance kernels, vector storage; exact
> (Flat) search, dataset I/O and the `vectorforge gen-data` / `ground-truth` CLI; single-threaded
> HNSW approximate search ([docs/hnsw.md](docs/hnsw.md)); checksummed, crash-safe persistence with
> memory-mapped loading and the `build` / `search` / `info` / `verify` CLI
> ([docs/storage-format.md](docs/storage-format.md)); AVX2 + FMA distance kernels selected at runtime
> ([docs/simd.md](docs/simd.md)); a thread pool, parallel batch search and a fully thread-safe
> `Collection` with a fair reader/writer lock and concurrent HNSW insertion
> ([docs/concurrency.md](docs/concurrency.md)); an HTTP/JSON server over a catalog of named
> collections ([docs/http-api.md](docs/http-api.md)); Python bindings
> ([docs/python-api.md](docs/python-api.md)).

## Quick start (current API)

```cpp
#include <vectorforge/collection.hpp>

vf::CollectionConfig cfg;
cfg.dim = 3;
cfg.metric = vf::Metric::Cosine;
cfg.index = vf::IndexType::Hnsw;  // default; IndexType::Flat for exact search
auto col = vf::Collection::create(cfg).value();
if (vf::Status st = col->add(42, std::vector<float>{0.1F, 0.2F, 0.3F}); !st.ok()) {
  std::fprintf(stderr, "%s\n", st.to_string().c_str());
}
vf::SearchParams params;
params.k = 5;
params.ef_search = 64;  // HNSW beam width (recall/latency trade-off)
auto hits = col->search(std::vector<float>{0.1F, 0.2F, 0.25F}, params).value();  // ascending distance

if (vf::Status st = col->save("docs.vfidx"); !st.ok()) {  // atomic write
  std::fprintf(stderr, "%s\n", st.to_string().c_str());
}
auto reopened = vf::Collection::load("docs.vfidx").value();  // vectors memory-mapped
```

```bash
vectorforge gen-data --n 100000 --dim 128 --seed 1 --format fvecs --out base.fvecs
vectorforge gen-data --n 1000 --dim 128 --seed 2 --format fvecs --out queries.fvecs
vectorforge ground-truth --base base.fvecs --queries queries.fvecs --metric l2 --k 100 --out gt
vectorforge build  --input base.fvecs --metric l2 --index hnsw --M 16 --ef-construction 200 --out idx.vfidx
vectorforge info   idx.vfidx
vectorforge verify idx.vfidx
vectorforge search --index idx.vfidx --queries queries.fvecs --k 10 --ef 100 --gt gt
```

```bash
vectorforge serve --data-dir ./data &          # 127.0.0.1:8080; docker compose up works too
curl -X POST localhost:8080/v1/collections -H 'Content-Type: application/json' \
  -d '{"name": "docs", "dim": 3, "metric": "cosine"}'
curl -X POST localhost:8080/v1/collections/docs/vectors -H 'Content-Type: application/json' \
  -d '{"vectors": [{"id": 42, "vector": [0.1, 0.2, 0.3]}]}'
curl -X POST localhost:8080/v1/collections/docs/search -H 'Content-Type: application/json' \
  -d '{"vector": [0.1, 0.2, 0.25], "k": 5}'
curl -X POST localhost:8080/v1/collections/docs/snapshot   # durable from here on
```

```python
import numpy as np, vectorforge as vf        # pip install ./python

index = vf.Index(dim=128, metric="cosine", M=16, ef_construction=200)
index.add(np.random.default_rng(0).standard_normal((10_000, 128), dtype=np.float32))
labels, distances = index.search(np.ones((3, 128), dtype=np.float32), k=5)  # (3, 5) arrays
index.save("docs.vfidx")
```

A batch search from Python runs at the speed of the C++ call
([results](benchmarks/results/2026-09-17_ryzen7-4800h_msvc-release_phase8/README.md)).
The HTTP API ([docs/http-api.md](docs/http-api.md), [OpenAPI](docs/openapi.yaml)) adds about
0.2 ms per query on loopback, and binary bulk ingestion is ~23× faster than JSON
([results](benchmarks/results/2026-09-17_ryzen7-4800h_msvc-release_phase7/README.md)).

The pipeline reads TEXMEX (SIFT-format) `.fvecs` vectors and `.ivecs` ground truth
(`--gt groundtruth.ivecs`); it is tested with generated files of that format, not with the SIFT1M
download itself.

Distance kernels use AVX2 + FMA when the CPU supports them and scalar code otherwise; set
`VF_SIMD=scalar` to force the scalar tier (for example to compare results or timings with the same
binary). On the development laptop (Ryzen 7 4800H, MSVC, one thread) AVX2 made HNSW queries over
100 000 × 128-d vectors 1.88× faster at recall 0.991 and Flat scans 2.97× faster
([results](benchmarks/results/2026-09-14_ryzen7-4800h_msvc-release_phase5/README.md)).

`vf::Collection` may be shared between threads: searches run in parallel, mutations are serialised,
and `compact()` rebuilds without blocking searches; HNSW rows are linked into the graph while
searches run. Pass a `vf::ThreadPool` to `search_batch` / `add_batch` (or `--threads` to
`ground-truth` / `build`) to use several cores. On the same laptop, batch HNSW search reached 4.8×
at 16 threads and HNSW construction 6.2× at 8 threads (100 000 × 128-d, recall unchanged), and
searches during ingestion kept a 0.084 ms median
([6a](benchmarks/results/2026-09-16_ryzen7-4800h_msvc-release_phase6a/README.md),
[6b](benchmarks/results/2026-09-16_ryzen7-4800h_msvc-release_phase6b/README.md)).

## Goals

- Exact k-NN and HNSW approximate search under L2, inner product and cosine metrics
- Scalar reference kernels plus AVX2/FMA kernels selected at runtime
- Concurrent searches with documented thread-safety guarantees
- Crash-safe, checksummed, versioned persistence with mmap-backed vector storage
- HTTP/JSON API, CLI and Python (pybind11/NumPy) frontends
- Correctness verified against brute-force ground truth; sanitizer-clean test suite
- Reproducible benchmarks with full environment metadata

See [docs/DESIGN.md](docs/DESIGN.md) for architecture, algorithms, concurrency arguments, storage
format, API design, test and benchmark methodology, and the phased implementation plan.

## Performance

Measured on the development laptop (AMD Ryzen 7 4800H, 8 cores / 16 threads, 15.4 GB, Windows 11,
MSVC release build, AVX2) with seeded synthetic data; methodology and what was not run:
[docs/benchmarking.md](docs/benchmarking.md). The tables are generated from the committed result
files by `benchmarks/scripts/make_readme_tables.py`.

<!-- BENCHMARKS:BEGIN -->
<!-- generated by benchmarks/scripts/make_readme_tables.py from benchmarks/results/2026-09-17_ryzen7-4800h_msvc-release_suite -->

#### Exact search (queries/s, k = 10)

| N | tier | threads | d=128 | d=384 | d=768 | d=1536 |
|---|---|---|---|---|---|---|
| 10 000 | scalar | 1 | 1 290.0 | 368.5 | 182.0 | 90.0 |
| 10 000 | scalar | 16 | 11 849.5 | 3 138.7 | 1 436.9 | 640.5 |
| 10 000 | avx2 | 1 | 4 135.8 | 1 374.2 | 663.3 | 315.8 |
| 10 000 | avx2 | 16 | 29 179.9 | 6 285.4 | 1 987.6 | 744.0 |
| 100 000 | scalar | 1 | 128.9 | 37.3 | 18.1 | 9.1 |
| 100 000 | scalar | 16 | 825.4 | 262.0 | 118.4 | 59.9 |
| 100 000 | avx2 | 1 | 306.2 | 129.4 | 65.6 | 34.0 |
| 100 000 | avx2 | 16 | 902.2 | 279.9 | 135.7 | 64.2 |
| 1 000 000 | scalar | 1 | 13.2 | 3.7 | 1.8 | — |
| 1 000 000 | scalar | 16 | 40.5 | 11.5 | 5.5 | — |
| 1 000 000 | avx2 | 1 | 38.4 | 12.6 | 6.8 | — |
| 1 000 000 | avx2 | 16 | 96.8 | 31.3 | 12.4 | — |

#### HNSW parameter sweep (100K × 128-d, one thread)

| M | efC | k | build s (1 thread) | graph MiB | QPS @ recall≥0.90 | QPS @ ≥0.95 | QPS @ ≥0.99 |
|---|---|---|---|---|---|---|---|
| 8 | 100 | 10 | 9.0 | 10 | 14 112 (ef 64) | 8 773 (ef 128) | 3 834 (ef 512) |
| 8 | 200 | 10 | 13.7 | 10 | 13 745 (ef 64) | 8 502 (ef 128) | 5 521 (ef 256) |
| 8 | 400 | 10 | 22.9 | 10 | 14 208 (ef 64) | 8 840 (ef 128) | 5 778 (ef 256) |
| 16 | 100 | 10 | 13.1 | 18 | 15 737 (ef 32) | 10 213 (ef 64) | 7 130 (ef 128) |
| 16 | 200 | 1 | 17.6 | 18 | 15 914 (ef 32) | 15 914 (ef 32) | 10 332 (ef 64) |
| 16 | 200 | 10 | 18.2 | 18 | 15 642 (ef 32) | 10 349 (ef 64) | 10 349 (ef 64) |
| 16 | 200 | 100 | 17.9 | 18 | 7 873 (ef 10) | 7 873 (ef 10) | 7 873 (ef 10) |
| 16 | 400 | 10 | 27.9 | 18 | 15 832 (ef 32) | 10 363 (ef 64) | 10 363 (ef 64) |
| 32 | 100 | 10 | 27.3 | 34 | 16 689 (ef 16) | 11 398 (ef 32) | 11 398 (ef 32) |
| 32 | 200 | 10 | 32.2 | 34 | 16 438 (ef 16) | 11 399 (ef 32) | 11 399 (ef 32) |
| 32 | 400 | 10 | 43.3 | 34 | 16 500 (ef 16) | 11 436 (ef 32) | 11 436 (ef 32) |
| 48 | 100 | 10 | 45.6 | 50 | 16 994 (ef 10) | 13 060 (ef 16) | 10 000 (ef 32) |
| 48 | 200 | 10 | 50.7 | 50 | 16 883 (ef 10) | 13 467 (ef 16) | 9 867 (ef 32) |
| 48 | 400 | 10 | 63.9 | 50 | 17 004 (ef 10) | 13 507 (ef 16) | 9 801 (ef 32) |

#### HNSW scale (M = 16, efC = 200, 16 build threads)

| N | d | build s (16 threads) | graph MiB | peak RSS GiB | recall@10 ef=64 | ef for ≥0.95 | unreachable |
|---|---|---|---|---|---|---|---|
| 10 000 | 128 | 0.1 | 9 | 0.03 | 1.0000 | 10 | 0 |
| 10 000 | 384 | 0.3 | 9 | 0.05 | 1.0000 | 10 | 0 |
| 10 000 | 768 | 0.8 | 10 | 0.08 | 1.0000 | 10 | 0 |
| 10 000 | 1536 | 1.8 | 10 | 0.13 | 1.0000 | 10 | 0 |
| 100 000 | 128 | 2.5 | 22 | 0.13 | 0.9902 | 64 | 0 |
| 100 000 | 384 | 5.9 | 22 | 0.32 | 0.9762 | 64 | 1 |
| 100 000 | 768 | 12.5 | 22 | 0.61 | 0.9698 | 64 | 1 |
| 100 000 | 1536 | 23.4 | 22 | 1.18 | 0.9680 | 64 | 4 |
| 1 000 000 | 128 | 96.8 | 173 | 1.21 | 0.7908 | 256 | 108 |
| 1 000 000 | 384 | 182.2 | 173 | 3.12 | 0.7098 | 512 | 583 |
| 1 000 000 | 768 | 402.6 | 175 | 5.98 | 0.6732 | 512 | 1063 |

#### Thread scaling (speedup vs one thread)

| workload | 1 T | 2 T | 4 T | 8 T | 12 T | 16 T |
|---|---|---|---|---|---|---|
| build-hnsw-100k-d128 | 1.00× | 1.96× | 3.68× | 5.71× | — | 7.36× |
| search-flat-100k-d128 | 1.00× | 1.88× | 3.02× | 4.71× | 6.13× | 2.70× |
| search-hnsw-100k-d128 | 1.00× | 1.91× | 3.14× | 4.15× | 4.43× | 4.71× |

#### Search latency during ingestion

| mode / writer threads | vectors/s | idle p50 ms | busy p50 ms | busy p99.9 ms | busy max ms |
|---|---|---|---|---|---|
| coarse-w1 | 3 738 | 0.067 | 2.278 | 2.89 | 3.32 |
| concurrent-w1 | 4 388 | 0.072 | 0.087 | 0.30 | 2.63 |
| concurrent-w4 | 15 508 | 0.067 | 0.098 | 0.42 | 2.51 |

Raw data, environment and interpretation: [benchmarks/results/2026-09-17_ryzen7-4800h_msvc-release_suite](benchmarks/results/2026-09-17_ryzen7-4800h_msvc-release_suite/README.md).
<!-- BENCHMARKS:END -->

Earlier focused measurements: SIMD
([phase 5](benchmarks/results/2026-09-14_ryzen7-4800h_msvc-release_phase5/README.md)), locking
and thread scaling ([6a](benchmarks/results/2026-09-16_ryzen7-4800h_msvc-release_phase6a/README.md),
[6b](benchmarks/results/2026-09-16_ryzen7-4800h_msvc-release_phase6b/README.md)), HTTP overhead
([7](benchmarks/results/2026-09-17_ryzen7-4800h_msvc-release_phase7/README.md)), Python overhead
([8](benchmarks/results/2026-09-17_ryzen7-4800h_msvc-release_phase8/README.md)).

## Building

Requirements: CMake ≥ 3.25, Ninja, and a C++20 compiler (MSVC 19.4x+, GCC 13+, Clang 17+).
GoogleTest, Google Benchmark, CLI11, cpp-httplib and nlohmann/json (server) and pybind11 (in-tree
Python module) are fetched automatically, pinned by version and SHA-256; the core library has no
third-party dependencies.

### Windows (MSVC)

```powershell
. .\tools\dev-env.ps1                # enters the VS x64 developer shell (MSVC + bundled CMake/Ninja)
cmake --preset msvc-release
cmake --build --preset msvc-release
ctest --preset msvc-release
```

Other presets: `msvc-debug`, `msvc-asan` (AddressSanitizer, optimised with assertions), `mingw-release` (compile check).

### Linux

```bash
cmake --preset linux-gcc-release
cmake --build --preset linux-gcc-release
ctest --preset linux-gcc-release
```

Sanitizer presets: `linux-clang-asan-ubsan`, `linux-clang-tsan`, `linux-clang-fuzz` (libFuzzer; run `tools/fuzz_index_reader.sh out/build/linux-clang-fuzz 600`).

### Useful CMake options

| Option | Default | Meaning |
|---|---|---|
| `VF_BUILD_TESTS` | ON | GoogleTest suites |
| `VF_BUILD_BENCHMARKS` | ON | Google Benchmark micro benchmarks and `vf_bench` (needs `VF_BUILD_CLI`) |
| `VF_SANITIZE` | empty | `address`, `address;undefined`, or `thread` |
| `VF_WARNINGS_AS_ERRORS` | OFF (ON in presets) | treat warnings as errors |
| `VF_ENABLE_AVX2` | ON | compile AVX2 kernels for runtime dispatch (x86-64) |
| `VF_NATIVE` | OFF | build for the host CPU (benchmark comparisons only) |
| `VF_ENABLE_ASSERTS` | OFF | keep internal assertions in optimised builds |
| `VF_BUILD_FUZZERS` | OFF | libFuzzer targets (Clang; preset `linux-clang-fuzz`) |
| `VF_BUILD_SERVER` | ON | HTTP server library and `vectorforge serve` |
| `VF_BUILD_PYTHON` | OFF | build the Python module in-tree (`pip install ./python` is the usual route) |

## Repository layout

```
include/vectorforge/   public headers
src/                   library implementation (internal headers)
apps/cli/              vectorforge command-line tool (including `serve`)
python/                Python package (pybind11) and its pytest suite
tests/                 GoogleTest suites
benchmarks/            micro (Google Benchmark) and macro (vf_bench) benchmarks, committed results
cmake/                 build modules (warnings, sanitizers, SIMD flags, dependencies)
examples/              curl walkthrough, Python quickstart
tools/                 developer scripts (golden files, fuzzing, ISA leak check, OpenAPI check)
docs/                  design document, component docs, OpenAPI spec, architecture decision records
```

## License

MIT — see [LICENSE](LICENSE).
