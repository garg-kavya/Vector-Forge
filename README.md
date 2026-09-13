# VectorForge

VectorForge is a C++20 vector similarity search engine built from first principles: exact
(brute-force) and approximate (HNSW) k-nearest-neighbour search over float32 vectors, with SIMD
distance kernels, a thread pool, a versioned on-disk format with memory-mapped vectors, an HTTP API
and Python bindings.

> **Status:** early development — Phases 0–3 of the [engineering design](docs/DESIGN.md) are
> complete: toolchain; core types, deterministic RNG, scalar distance kernels, vector storage; exact
> (Flat) search, dataset I/O and the `vectorforge gen-data` / `ground-truth` CLI; single-threaded
> HNSW approximate search ([docs/hnsw.md](docs/hnsw.md)). Persistence, SIMD kernels, concurrency,
> HTTP and Python are *planned*, not implemented.

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
```

```bash
vectorforge gen-data --n 100000 --dim 128 --seed 1 --out base.npy
vectorforge gen-data --n 1000 --dim 128 --seed 2 --out queries.npy
vectorforge ground-truth --base base.npy --queries queries.npy --metric l2 --k 100 --out gt
```

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

No end-to-end performance results are published yet. Numbers will appear here only after the
benchmark suite has been run, with the raw result files committed under `benchmarks/results/` and the
methodology described in the design document (§16). Component-level baselines (kernels, top-k
selection, exact search, a single-configuration HNSW recall/latency sweep) are recorded, with their
environment, in `benchmarks/results/`.

## Building

Requirements: CMake ≥ 3.25, Ninja, and a C++20 compiler (MSVC 19.4x+, GCC 13+, Clang 17+).
GoogleTest, Google Benchmark and CLI11 are fetched automatically (pinned by version and SHA-256); the
core library has no third-party dependencies.

### Windows (MSVC)

```powershell
. .\tools\dev-env.ps1                # enters the VS x64 developer shell (MSVC + bundled CMake/Ninja)
cmake --preset msvc-release
cmake --build --preset msvc-release
ctest --preset msvc-release
```

Other presets: `msvc-debug`, `msvc-asan` (AddressSanitizer), `mingw-release` (compile check).

### Linux

```bash
cmake --preset linux-gcc-release
cmake --build --preset linux-gcc-release
ctest --preset linux-gcc-release
```

Sanitizer presets: `linux-clang-asan-ubsan`, `linux-clang-tsan`.

### Useful CMake options

| Option | Default | Meaning |
|---|---|---|
| `VF_BUILD_TESTS` | ON | GoogleTest suites |
| `VF_BUILD_BENCHMARKS` | ON | Google Benchmark micro benchmarks and `vf_bench` (needs `VF_BUILD_CLI`) |
| `VF_SANITIZE` | empty | `address`, `address;undefined`, or `thread` |
| `VF_WARNINGS_AS_ERRORS` | OFF (ON in presets) | treat warnings as errors |
| `VF_ENABLE_AVX2` | ON | compile AVX2 kernels for runtime dispatch |
| `VF_NATIVE` | OFF | build for the host CPU (benchmark comparisons only) |

## Repository layout

```
include/vectorforge/   public headers
src/                   library implementation (internal headers)
apps/cli/              vectorforge command-line tool
tests/                 GoogleTest suites
benchmarks/            micro (Google Benchmark) and macro (vf_bench) benchmarks, committed results
cmake/                 build modules (warnings, sanitizers, SIMD flags, dependencies)
tools/                 developer scripts
docs/                  design document and architecture decision records
```

## License

MIT — see [LICENSE](LICENSE).
