# VectorForge

VectorForge is a C++20 vector similarity search engine built from first principles: exact
(brute-force) and approximate (HNSW) k-nearest-neighbour search over float32 vectors, with AVX2
distance kernels chosen at runtime, concurrent insertion and search, a checksummed on-disk format
with memory-mapped vectors, an HTTP/JSON server, a CLI and Python bindings.

Status: all phases of the [engineering design](docs/DESIGN.md) are implemented (version 0.1.0,
not yet released). It is a single-node engine developed and measured on one laptop; read
[Limitations](#limitations) before relying on it.

## Contents

- [Quick start](#quick-start): [C++](#c) · [CLI](#cli) · [HTTP](#http) · [Python](#python)
- [Architecture](#architecture)
- [Design highlights](#design-highlights)
- [Performance](#performance)
- [Limitations](#limitations)
- [Building](#building) · [Repository layout](#repository-layout) · [Documentation](#documentation)

## Quick start

### C++

```cpp
#include <vectorforge/vectorforge.hpp>

vf::CollectionConfig cfg;
cfg.dim = 3;
cfg.metric = vf::Metric::Cosine;
cfg.index = vf::IndexType::Hnsw;  // default; IndexType::Flat for exact search
auto col = vf::Collection::create(cfg).value();

std::vector<vf::ExternalId> ids{1, 2};
std::vector<float> rows{0.1F, 0.2F, 0.3F, 0.3F, 0.2F, 0.1F};
vf::ThreadPool pool(4);                                   // optional
if (auto added = col->add_batch(ids, rows, {}, &pool); !added.ok()) {
  std::fprintf(stderr, "%s\n", added.status().to_string().c_str());
}

vf::SearchParams params;
params.k = 5;
params.ef_search = 64;  // HNSW beam width: recall/latency trade-off
auto hits = col->search(std::vector<float>{0.1F, 0.2F, 0.25F}, params).value();  // ascending distance

if (vf::Status st = col->save("docs.vfidx"); !st.ok()) {  // atomic write
  std::fprintf(stderr, "%s\n", st.to_string().c_str());
}
auto reopened = vf::Collection::load("docs.vfidx").value();  // vectors memory-mapped
```

As a package: `find_package(vectorforge REQUIRED)` and link `vectorforge::vectorforge` after
`cmake --install`; see [examples/cpp/quickstart.cpp](examples/cpp/quickstart.cpp).

### CLI

```bash
vectorforge gen-data --n 100000 --dim 128 --seed 1 --format fvecs --out base.fvecs
vectorforge gen-data --n 1000 --dim 128 --seed 2 --format fvecs --out queries.fvecs
vectorforge ground-truth --base base.fvecs --queries queries.fvecs --metric l2 --k 100 --out gt
vectorforge build  --input base.fvecs --metric l2 --index hnsw --M 16 --ef-construction 200 --out idx.vfidx
vectorforge info   idx.vfidx
vectorforge verify idx.vfidx
vectorforge search --index idx.vfidx --queries queries.fvecs --k 10 --ef 100 --gt gt
```

The pipeline reads TEXMEX (SIFT-format) `.fvecs`/`.ivecs` and NumPy `.npy` files.

### HTTP

```bash
vectorforge serve --data-dir ./data &          # 127.0.0.1:8080; or: docker compose up --build
curl -X POST localhost:8080/v1/collections -H 'Content-Type: application/json' \
  -d '{"name": "docs", "dim": 3, "metric": "cosine"}'
curl -X POST localhost:8080/v1/collections/docs/vectors -H 'Content-Type: application/json' \
  -d '{"vectors": [{"id": 42, "vector": [0.1, 0.2, 0.3]}]}'
curl -X POST localhost:8080/v1/collections/docs/search -H 'Content-Type: application/json' \
  -d '{"vector": [0.1, 0.2, 0.25], "k": 5}'
curl -X POST localhost:8080/v1/collections/docs/snapshot   # durable from here on
curl localhost:8080/metrics                                  # Prometheus
```

Reference: [docs/http-api.md](docs/http-api.md), [OpenAPI](docs/openapi.yaml),
[walkthrough](examples/http/curl_examples.sh).

### Python

```python
import numpy as np, vectorforge as vf        # pip install ./python

index = vf.Index(dim=128, metric="cosine", M=16, ef_construction=200)
index.add(np.random.default_rng(0).standard_normal((10_000, 128), dtype=np.float32))
labels, distances = index.search(np.ones((3, 128), dtype=np.float32), k=5)  # (3, 5) arrays
index.save("docs.vfidx")
```

Reference: [docs/python-api.md](docs/python-api.md), [quickstart](examples/python/quickstart.py).

## Architecture

```text
 frontends     vectorforge CLI · HTTP server (vf_server) · Python module · benchmarks
 ─────────────────────── public API: include/vectorforge/*.hpp ───────────────────────
 collection    Catalog (named collections, snapshots) · Collection (thread-safe façade)
 index         FlatBackend (blocked exact scan) · HnswBackend (graph, insert, search, validator)
 search        bounded heaps · visited sets · pooled search contexts
 storage       chunked VectorStore with mmap base · IdMap · tombstones · .vfidx format · CRC-32C
 simd          CPU feature detection · kernel table (scalar | AVX2+FMA)
 concurrency   ThreadPool · fair reader/writer lock · striped mutexes
 core          Status/Result · types · validation · RNG
```

The core library has no third-party dependencies. Details: [docs/architecture.md](docs/architecture.md).

## Design highlights

- **HNSW from the paper**, with heuristic neighbour selection, orphan repair, deterministic levels
  derived from `(seed, id)`, tombstones kept as navigation hubs, and a graph validator used by every
  build test ([docs/hnsw.md](docs/hnsw.md)).
- **Runtime SIMD dispatch**: one portable binary; AVX2+FMA kernels live in a single object file
  with its own flags, and a CI check proves no AVX instruction leaks elsewhere
  ([docs/simd.md](docs/simd.md), [ADR-0002](docs/adr/0002-runtime-simd-dispatch.md)).
- **Concurrent insertion with lock-free readers**: rows are appended in short exclusive sections
  and linked into the graph under a shared lock with atomic link lists and striped writer locks;
  searches keep a 0.08 ms median during ingestion instead of 2.3 ms, and builds run 6.2× faster on
  8 cores with the same recall ([docs/concurrency.md](docs/concurrency.md),
  [ADR-0003](docs/adr/0003-concurrent-hnsw-insert.md)). A fair reader/writer lock replaced
  `std::shared_mutex` after it was measured starving searches for 0.7 s.
- **Untrusted files**: every `.vfidx` section is checksummed and structurally validated; loads are
  fuzzed; saves are atomic and crash-tested at every step; snapshots are generation files behind an
  atomically replaced manifest ([docs/storage-format.md](docs/storage-format.md),
  [ADR-0005](docs/adr/0005-generation-snapshots.md)).
- **Errors as values**: `Status`/`Result` with stable codes mapped to HTTP status codes and Python
  exceptions; allocation failures keep documented guarantees, verified by injecting a failure
  into every allocation of every insert ([ADR-0001](docs/adr/0001-status-result-errors.md)).
- **Measured, not assumed**: every number below comes from a committed result file with its
  environment; ablations (prefetch, neighbour selection, accumulators, visited sets, lock
  designs) are kept, including the ones that did not help.

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

Reading the tables: exact search is memory-bound beyond a few threads once the data outgrows the
cache; thread-scaling points above 12 threads are noisy on this desktop machine (see the suite
README); HNSW needs a larger `ef_search` as collections grow (recall 0.79 at ef = 64 but 0.99 at
ef = 512 for 1 M × 128-d); the data is synthetic, and no real-dataset runs are included yet.

Earlier focused measurements: SIMD
([phase 5](benchmarks/results/2026-09-14_ryzen7-4800h_msvc-release_phase5/README.md)), locking
and thread scaling ([6a](benchmarks/results/2026-09-16_ryzen7-4800h_msvc-release_phase6a/README.md),
[6b](benchmarks/results/2026-09-16_ryzen7-4800h_msvc-release_phase6b/README.md)), HTTP overhead
([7](benchmarks/results/2026-09-17_ryzen7-4800h_msvc-release_phase7/README.md)), Python overhead
([8](benchmarks/results/2026-09-17_ryzen7-4800h_msvc-release_phase8/README.md)).

## Limitations

- **Single node, single machine measurements.** No replication, sharding or distributed search.
  All numbers come from one laptop with synthetic data; real datasets (SIFT1M, GloVe, embedding
  sets) are supported by the tools but were not downloaded or measured.
- **Durability is explicit.** Inserts are in memory until a snapshot (`save`, `POST …/snapshot`,
  `--snapshot-on-exit`); there is no write-ahead log.
- **Memory.** Vectors, ids and the graph are kept in RAM (vectors can be memory-mapped after a
  load, but new inserts go to the heap). 1M × 1536-d did not fit the 16 GB development machine
  alongside the benchmark's own copies.
- **Deletes are tombstones** until `compact()` rebuilds the collection; heavy delete workloads
  need periodic compaction.
- **Float32 only**; no quantisation, product codes, filtering by metadata or hybrid search.
- **Platforms.** Tested on Windows (MSVC, MinGW) and Linux (GCC, Clang) on x86-64. macOS and ARM
  builds are not tested; the AVX2 tier is x86-64 only (other CPUs use the scalar kernels).
- **HTTP.** No TLS (use a proxy), a single optional bearer token, no per-user authorisation or
  rate limiting. JSON ingestion is ~23× slower than the binary endpoint.
- **Level B details.** A parallel build is not bit-identical to a serial one; a rare allocation
  failure while linking leaves a tombstoned row; `add()` of a single vector still takes the
  exclusive lock.
- **Release artifacts** (wheels, image, archives) are produced by the release workflow; the Docker
  image and wheels for Linux were built only in CI, not on the development machine.

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
cmake --install out/build/msvc-release --prefix C:/opt/vectorforge
```

Other presets: `msvc-debug`, `msvc-asan` (AddressSanitizer, optimised with assertions),
`mingw-release`.

### Linux

```bash
cmake --preset linux-gcc-release
cmake --build --preset linux-gcc-release
ctest --preset linux-gcc-release
```

Sanitizer presets: `linux-clang-asan-ubsan`, `linux-clang-tsan`, `linux-clang-fuzz` (libFuzzer;
`tools/fuzz_index_reader.sh` and `tools/fuzz_json_request.sh`). Testing guide:
[docs/testing.md](docs/testing.md).

### Useful CMake options

| Option | Default | Meaning |
|---|---|---|
| `VF_BUILD_TESTS` | ON | GoogleTest suites |
| `VF_BUILD_BENCHMARKS` | ON | Google Benchmark micro benchmarks and `vf_bench` (needs `VF_BUILD_CLI`) |
| `VF_BUILD_CLI` | ON | the `vectorforge` tool |
| `VF_BUILD_SERVER` | ON | HTTP server library and `vectorforge serve` |
| `VF_BUILD_PYTHON` | OFF | build the Python module in-tree (`pip install ./python` is the usual route) |
| `VF_SANITIZE` | empty | `address`, `address;undefined`, or `thread` |
| `VF_WARNINGS_AS_ERRORS` | OFF (ON in presets) | treat warnings as errors |
| `VF_ENABLE_AVX2` | ON | compile AVX2 kernels for runtime dispatch (x86-64) |
| `VF_NATIVE` | OFF | build for the host CPU (benchmark comparisons only) |
| `VF_ENABLE_ASSERTS` | OFF | keep internal assertions in optimised builds |
| `VF_BUILD_FUZZERS` | OFF | libFuzzer targets (Clang; preset `linux-clang-fuzz`) |
| `VF_USE_SYSTEM_DEPS` | OFF | use installed packages instead of fetching dependencies |

Runtime settings (environment variables, server options, logging, metrics):
[docs/configuration.md](docs/configuration.md).

## Repository layout

```text
include/vectorforge/   public headers (installed)
src/                   library implementation; src/server is the HTTP server
apps/cli/              vectorforge command-line tool (including `serve`)
python/                Python package (pybind11) and its pytest suite
tests/                 GoogleTest suites, fuzz targets, install test
benchmarks/            micro and macro benchmarks, suite configs and scripts, committed results
examples/              C++ quickstart, curl walkthrough, Python quickstart
cmake/                 build modules (warnings, sanitizers, SIMD flags, dependencies, package config)
tools/                 developer scripts (golden files, fuzzing, ISA leak check, OpenAPI check,
                       datasets, release packaging)
docs/                  design document, component docs, OpenAPI spec, ADRs
```

## Documentation

[Design](docs/DESIGN.md) · [Architecture](docs/architecture.md) · [HNSW](docs/hnsw.md) ·
[SIMD](docs/simd.md) · [Concurrency](docs/concurrency.md) ·
[Storage format](docs/storage-format.md) · [HTTP API](docs/http-api.md) ·
[Python API](docs/python-api.md) · [Configuration](docs/configuration.md) ·
[Testing](docs/testing.md) · [Benchmarking](docs/benchmarking.md) · [ADRs](docs/adr/) ·
[Changelog](CHANGELOG.md)

## License

MIT — see [LICENSE](LICENSE).
