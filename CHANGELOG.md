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

- Phase 5: SIMD. AVX2 + FMA kernels (`dot`, `l2sq`, `norm2`, 1-to-N) in variants with four or one
  accumulators and a scalar or masked-load tail, compiled as an isolated object library
  (`vf_simd_avx2`); the `avx2` tier uses four accumulators with a scalar tail. Runtime tier selection
  from CPUID/XGETBV with the `VF_SIMD` override (`auto`, `scalar`, `avx2`); invalid or unsupported
  requests are reported by the new `vf::simd_status()` and by `Collection::create/load`,
  `vf::distance`, `vf::normalize` and the CLI instead of being downgraded. `vectorforge info` prints
  the tier. `HnswSearchOptions::prefetch` (on by default) prefetches neighbour rows during HNSW
  searches. `tools/check_isa_leak.py` (objdump/dumpbin) verifies that AVX instructions stay in the
  AVX2 object. Tests: every kernel table across dims and offsets, bit-identical alignment
  independence, symmetry and batch/pairwise equality, tier selection against synthetic CPU features,
  CTest runs with `VF_SIMD` set to scalar/avx2/invalid, per-tier golden HNSW fingerprints, prefetch
  result invariance. Benchmarks: kernel variants, dispatch-level experiment (table vs direct vs
  inlined), `vf_bench --simd` and `--prefetch` A/B on saved index files (`--index-file` for sweeps).
  CI runs the full test suite on both tiers and the ISA leak check on GCC, Clang and MSVC.
  `docs/simd.md`, ADR-0002.

- Phase 6a: thread pool and Level A concurrency. Public `vf::ThreadPool` (`std::jthread` workers,
  `submit`/`try_submit` futures, `parallel_for` with caller participation, inline nested calls,
  first-exception propagation, idempotent `shutdown`). `vf::Collection` is now fully thread-safe:
  a fair reader/writer lock (`detail::FairSharedMutex`: writer preference plus a reader quota),
  a writer mutex, time-bounded (2 ms) `add_batch` lock sections, state held in a `shared_ptr`.
  `Collection::compact()` drops removed rows (and moves mapped vectors to the heap) while searches
  continue. `search_batch` runs queries on an optional pool; `add_batch` normalises rows on it.
  CLI `--threads` for `ground-truth` and `build`. Tests: thread pool, concurrent search against
  writers (Flat and HNSW), searches during compaction, a writer/compactor/saver/reader stress test
  checked against a model (label `stress`), compaction. `vf_bench` scenarios `threads` and
  `ingest`. Nightly workflow repeating the stress tests 100 times under TSan and in release.
  `docs/concurrency.md`.

- Phase 6b: concurrent HNSW insertion (Level B), the new default
  (`CollectionConfig::concurrency` / `LoadOptions::concurrency`: `Concurrent` or `Coarse`).
  `add_batch` appends rows and allocates graph nodes in short exclusive sections and links them
  while searches run, in parallel on an optional pool. Atomic link lists read without locks,
  striped writer mutexes, an atomic entry point guarded for top-level insertions, per-insertion
  scratch pools, atomic build statistics; saves wait for the link section in progress. A row whose
  linking fails with `std::bad_alloc` is tombstoned and its id restored. `vf::to_string` /
  `vf::parse_concurrency`. Tests: serial Concurrent build equals the Coarse build, parallel builds
  (validator, reachability, recall), upserts, entry-point races, searches/removals/saves during
  parallel ingestion (stress label), allocation failure injection in both modes. `vf_bench`
  `build` scenario and `--concurrency` / `--writer-threads` for `ingest`. ADR-0003.

- Phase 7: HTTP API. `vf::Catalog` (named collections under a data directory: create, get, list,
  drop with deferred file deletion, generation snapshots, restart recovery, crash-safe creation
  and drops). Server library `vf_server` (cpp-httplib 0.54.1, nlohmann/json 3.12.0, pinned) with
  16 routes (health, readiness, status, collections, vectors, binary bulk insert, search, batch
  search, stats, snapshot, compact), strict request decoding, stable JSON error codes and HTTP
  status mapping, request ids, optional bearer token (constant-time comparison), body/batch/k/ef/
  dim limits, JSON depth limit, exclusive port binding, graceful drain. `vectorforge serve`
  (signals and Windows console events, `--snapshot-on-exit`, `VF_API_KEY`, `--list-routes`).
  `docs/http-api.md`, `docs/openapi.yaml` (checked against the route table in CI),
  `examples/http/curl_examples.sh`, Dockerfile (non-root), docker-compose, Docker CI workflow.
  Tests: every route against the library, error mapping, limits, authentication, restart from a
  snapshot, drain on stop, drop while in use, concurrent clients, catalog recovery and damaged
  files; JSON request fuzz target (replay test everywhere, libFuzzer in CI). `vf_http_bench`.

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
- The default kernel tier is AVX2 on CPUs that support it (was scalar). Results differ from the
  scalar tier within floating-point rounding, so HNSW graphs built on different tiers can differ;
  the cosine golden fingerprint has a separate AVX2 value.
- `SimdLevel` and `to_string(SimdLevel)` moved to `vectorforge/simd_level.hpp` (still included by
  `vectorforge/simd.hpp`).
- `msvc-asan` builds RelWithDebInfo with `VF_ENABLE_ASSERTS=ON` (was Debug), so the allocation
  failure-injection tests also run under MSVC ASan.
- HNSW back-link shrinking uses an insertion sort that stays well-defined for NaN distances.
- `Collection::create` accepts `IndexType::Hnsw` (previously `FailedPrecondition`).
- `IndexBackend::search` may throw `std::bad_alloc` (growing HNSW search contexts).
- `FlatBackend` distance formulas are compiled without floating-point contraction.
- `Status` keeps its message behind a pointer so OK statuses never allocate.
- Builds record `-dirty` in the embedded git sha when configured from a modified working tree.
- `mingw-release` links the GCC runtime statically.
