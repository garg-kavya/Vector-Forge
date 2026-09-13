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

### Changed
- `Collection::create` accepts `IndexType::Hnsw` (previously `FailedPrecondition`).
- `IndexBackend::search` may throw `std::bad_alloc` (growing HNSW search contexts).
- `FlatBackend` distance formulas are compiled without floating-point contraction.
- `Status` keeps its message behind a pointer so OK statuses never allocate.
- Builds record `-dirty` in the embedded git sha when configured from a modified working tree.
- `mingw-release` links the GCC runtime statically.
