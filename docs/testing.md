# Testing

How VectorForge is tested and how to run the tests. Strategy: [DESIGN.md §15](DESIGN.md#15-testing-strategy-p).

## Running

```powershell
. .\tools\dev-env.ps1
cmake --preset msvc-release; cmake --build --preset msvc-release
ctest --preset msvc-release                      # everything
ctest --preset msvc-release -L http              # one label
$env:VF_SIMD = "scalar"; ctest --preset msvc-release   # the scalar kernel tier
```

```bash
cmake --preset linux-clang-tsan && cmake --build --preset linux-clang-tsan
ctest --preset linux-clang-tsan                  # ThreadSanitizer (needs vm.mmap_rnd_bits=28 on recent kernels)
ctest --preset linux-clang-tsan -L stress --repeat until-fail:100
pip install "./python[test]" && pytest python/tests
python -m pytest benchmarks/scripts/tests       # benchmark harness
```

## Test binaries and labels

| Binary / test | Label | Contents |
|---|---|---|
| `vf_unit_tests` | `unit` | status, types and config, checked math, RNG, aligned allocation, kernels on every tier, dispatch, normalisation, validation, vector store, id map, CRC-32C, binary I/O, mapped files, top-k, visited set, Flat backend against a double-precision reference, model-based Flat test, HNSW graph, level generator, neighbour selection, validator, tombstones, edge cases, determinism (golden fingerprints per tier), compaction, dataset I/O, recall, logging |
| `vf_alloc_tests` | `unit` | allocation-failure injection for every insert (Flat, HNSW Coarse, HNSW Concurrent); zero allocations on the search path |
| `vf_concurrency_tests` | `concurrency` | thread pool, concurrent reads, readers against writers, parallel HNSW builds (validator, recall, Coarse/Concurrent equality) |
| `vf_stress_tests` | `concurrency;stress` | writer/compactor/saver/readers checked against a model; searches, removals and saves during parallel ingestion |
| `vf_persistence_tests`, `fuzz.index_reader_replay` | `persistence` | round trips in every load mode, atomic saves with fault injection, truncation and bit-flip corruption, hostile values with valid checksums, golden files from an independent Python writer, fuzz corpus replay |
| `vf_index_integration_tests` | `integration` | HNSW recall against exact results (frozen thresholds), catalog lifecycle and recovery |
| `vf_integration_tests` | `integration` | CLI pipeline on generated SIFT-format files |
| `example.cpp_quickstart`, `install.consumer` | `integration` | the C++ quickstart in-tree, and built against an installed package with `find_package(vectorforge)` (skipped in sanitizer builds) |
| `vf_http_tests`, `fuzz.json_request_replay` | `http` | every route against the library, error mapping, limits, authentication, drain on stop, drop while in use, concurrent clients, metrics, access log; JSON decoder corpus replay |
| `simd.*` | `unit`, `persistence` | tier-sensitive tests under `VF_SIMD=scalar`, `avx2`, and invalid requests |
| `python.pytest` (with `-DVF_BUILD_PYTHON=ON`) | `python` | the pytest suite against the in-tree module |
| `python/tests` (pytest) | — | exactness against NumPy, recall, persistence, input conversion and zero-copy, errors, threads and GIL release |

## Oracles

- **Exact search**: a double-precision brute-force reference (`tests/support/brute_force_reference.hpp`).
- **HNSW**: the graph validator (invariants and per-level reachability) after every build; recall
  against exact results; golden graph fingerprints for determinism.
- **Stateful behaviour**: model-based tests (a `std::map` model driven by random operations).
- **Files**: golden files written by an independent Python implementation (`tools/make_golden.py`).
- **Untrusted input**: libFuzzer targets for the index reader and the HTTP request decoders
  (`tools/fuzz_index_reader.sh`, `tools/fuzz_json_request.sh`), run for 2 + 1 minutes in CI and
  10 minutes each nightly.

## Configurations run before every commit (development machine)

MSVC release/debug/ASan and VS 2022 toolset, MinGW GCC; in WSL: GCC 13 release/debug, Clang 18
release, ASan+UBSan, TSan; both kernel tiers on the release builds; the ISA leak check;
clang-tidy 18 on library, server and CLI sources; clang-format 19.1.5. CI
(`.github/workflows/`) runs the equivalent on GitHub-hosted runners, plus Python, Docker, fuzzing,
benchmark smoke and nightly stress jobs.

## Writing tests

- Put new cases next to the component's existing test file; register new files in
  `tests/CMakeLists.txt` with the right label.
- Temporary files go through `vf::test::ScopedTempDir`; synthetic data through
  `vf::test::ClusteredData` / `random_matrix` with fixed seeds.
- Concurrency tests must be meaningful under TSan: keep sizes small and assert invariants rather
  than timing.
