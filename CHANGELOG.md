# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses semantic versioning.

## [Unreleased]

### Added
- Phase 0: repository scaffolding, CMake build with presets (MSVC, MinGW, Linux GCC/Clang,
  sanitizers), pinned GoogleTest and Google Benchmark, warning and sanitizer modules,
  `tools/dev-env.ps1`, clang-format/clang-tidy configuration, CI and lint workflows,
  engineering design document.
- Phase 1: `Status`/`Result<T>` error handling; core types (`Metric`, `IndexType`, ids, `Neighbor`)
  and validated `HnswParams`/`CollectionConfig`; overflow-checked arithmetic; deterministic
  SplitMix64/xoshiro256** RNG with fully specified float conversions; 64-byte aligned allocation;
  scalar reference distance kernels (dot, squared L2, norm, 1-to-N) plus compiler-autovectorised
  comparison variants; x86 CPU feature detection (CPUID + XGETBV); kernel dispatch table (scalar);
  input validation, normalisation and public `vf::distance`/`vf::normalize`; chunked append-only
  `VectorStore` with stable row addresses; unit tests; Google Benchmark scalar kernel baseline.
- Phase 2: exact search. Public `vf::Collection` (Flat index): add, add_batch (all-or-nothing
  validation), upsert, remove (tombstones), get, contains, search, allocation-free search_into,
  search_batch with padding, stats with memory breakdown. `IdMap` with reserve/commit/rollback,
  `TombstoneSet`, `IndexBackend` interface and blocked, chunk-aware `FlatBackend`; bounded max-heap
  and min-heap with deterministic (distance, id) ordering. NumPy `.npy` (streaming reader/writer) and
  TEXMEX `.fvecs`/`.ivecs` I/O; tie-tolerant `recall_at_k`; seeded synthetic uniform and Gaussian
  mixture generators. `vectorforge` CLI (CLI11) with `gen-data` and `ground-truth`. Tests: double-
  precision brute-force reference, 10⁴-operation model-based tests, zero-allocation test binary,
  CLI pipeline integration tests and executable smoke tests. Benchmarks: top-k strategies and Flat
  search latency baselines.

- Phase 3: single-threaded HNSW. `IndexType::Hnsw` collections (now creatable) with `HnswBackend`:
  chunked graph storage (fixed-stride level 0, upper-level arena, `LinkView`), exact integer
  level generation derived from `(seed, id)`, greedy descent and beam search with an epoch-stamped
  `VisitedSet`, pooled `SearchContext`s, heuristic neighbour selection with `keep_pruned`,
  four-phase insertion with back-link shrinking and orphan repair, tombstone-aware queries.
  `HnswValidator` (invariants, per-level reachability, level histogram), canonical graph encoding.
  Tests: level generator, visited set, neighbour selection, graph storage, validator, edge cases,
  tombstones, model-based random operations, determinism with golden fingerprints, recall against
  Flat ground truth (integration), zero-allocation HNSW queries. Benchmarks: `bench_visited`,
  minimal `vf_bench` (build, ef_search sweep, recall, latency, distance computations, Flat
  comparison). `docs/hnsw.md`.

- Phase 4: persistence. `.vfidx` format 1.0 (`docs/storage-format.md`): checksummed (CRC-32C
  slice-by-8) header, section table and sections; canonical layout with page-aligned vectors.
  `Collection::save` (atomic temp + sync + rename, byte-identical re-saves) and
  `Collection::load(path, LoadOptions)` with heap or read-only mmap loading (`MappedFile`, Win32 and
  POSIX; inserts continue in heap chunks after the mapped base), `Verify` levels and complete
  validation of untrusted files. Generation snapshots with MANIFEST, garbage collection and a
  fault-injection hook. CLI `build`, `search` (latency, recall against npy or `.ivecs` ground
  truth), `info`, `verify`. Tests: CRC known answers, binary codecs, mapped files, atomic writes,
  round trips in every load mode, loaded collections evolving identically, truncation and
  every-byte bit-flip corruption suites, hostile values with valid checksums, crash injection for
  saves and snapshots, golden files from an independent Python writer (`tools/make_golden.py`),
  CLI pipeline on SIFT-format files. libFuzzer target for the reader (preset `linux-clang-fuzz`,
  corpus replay test everywhere). `vf_bench` storage scenarios (save, heap/mmap load).
  CI: Windows MSVC ASan job, fuzz smoke job, nightly 10-minute fuzz workflow.

- Fault-injection tests (`test_exception_safety`): every allocation of every insert fails in turn;
  failed inserts must leave collections unchanged and fully searchable. Concurrent-read test
  (`vf_concurrency_tests`) for the const-member thread-safety contract, clean under TSan.

### Fixed
- A failed or throwing backend `add()` left the appended row behind (tombstoned). For HNSW this
  desynchronised row ids from graph node ids, so later inserts aborted in debug builds and indexed
  the wrong vectors in release builds. Collections now undo the append (`VectorStore::pop_back`),
  giving `add` the documented strong guarantee; `IdMap::rollback_appended` was removed.
- `HnswBackend::add` could index past the end of its per-level scratch after an allocation failure
  between two scratch resizes.
- The `linux-clang-tsan` preset failed to link: the allocation-test binary replaces the global
  `operator new`, which ThreadSanitizer's runtime also defines. That binary is now skipped under TSan.

### Changed
- `msvc-asan` builds RelWithDebInfo with `VF_ENABLE_ASSERTS=ON` (was Debug), so the allocation
  failure-injection tests also run under MSVC ASan.
- HNSW back-link shrinking uses an insertion sort that stays well-defined for NaN distances.
- `Collection::create` accepts `IndexType::Hnsw` (previously `FailedPrecondition`).
- `IndexBackend::search` may throw `std::bad_alloc` (growing HNSW search contexts).
- `FlatBackend` distance formulas are compiled without floating-point contraction.
- `Status` keeps its message behind a pointer so OK statuses never allocate.
- Builds record `-dirty` in the embedded git sha when configured from a modified working tree.
- `mingw-release` links the GCC runtime statically.
