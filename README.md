# VectorForge

VectorForge is a C++20 vector similarity search engine built from first principles: exact
(brute-force) and approximate (HNSW) k-nearest-neighbour search over float32 vectors, with SIMD
distance kernels, a thread pool, a versioned on-disk format with memory-mapped vectors, an HTTP API
and Python bindings.

> **Status:** early development — Phase 0 (repository and toolchain) of the
> [engineering design](docs/DESIGN.md). Most features listed below are *planned*, not implemented.

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

No performance results are published yet. Numbers will appear here only after benchmarks have been
run, with the raw result files committed under `benchmarks/results/` and the methodology described
in the design document (§16).

## Building

Requirements: CMake ≥ 3.25, Ninja, and a C++20 compiler (MSVC 19.4x+, GCC 13+, Clang 17+).
GoogleTest and Google Benchmark are fetched automatically (pinned by version and SHA-256).

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
| `VF_BUILD_BENCHMARKS` | ON | Google Benchmark micro benchmarks |
| `VF_SANITIZE` | empty | `address`, `address;undefined`, or `thread` |
| `VF_WARNINGS_AS_ERRORS` | OFF (ON in presets) | treat warnings as errors |
| `VF_ENABLE_AVX2` | ON | compile AVX2 kernels for runtime dispatch |
| `VF_NATIVE` | OFF | build for the host CPU (benchmark comparisons only) |

## Repository layout

```
include/vectorforge/   public headers
src/                   library implementation (internal headers)
tests/                 GoogleTest suites
benchmarks/            micro (Google Benchmark) and, later, macro benchmarks
cmake/                 build modules (warnings, sanitizers, SIMD flags, dependencies)
tools/                 developer scripts
docs/                  design document and architecture decision records
```

## License

MIT — see [LICENSE](LICENSE).
