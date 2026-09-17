# Architecture

A map of the code as built. The reasoning behind it is in [DESIGN.md](DESIGN.md) (§4–§14); each
component has its own page.

```text
 ┌──────────────────────────────── frontends ─────────────────────────────────┐
 │ apps/cli (vectorforge)  src/server (vf_server)  python/ (_vectorforge)      │
 │                                   benchmarks/ (vf_bench, vf_micro_bench)    │
 └──────────────┬─────────────────────────┬─────────────────────┬─────────────┘
                │      public API: include/vectorforge/*.hpp    │
 ┌──────────────▼─────────────────────────▼─────────────────────▼─────────────┐
 │ collection/  Catalog · Collection (validation, ids, locks, compaction)      │
 │              index files · generation snapshots                             │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │ index/       IndexBackend ← FlatBackend | HnswBackend (graph, insert,       │
 │              search, validator, graph I/O)                                  │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │ search/      heaps · visited set · search contexts and their pool           │
 │ storage/     VectorStore (chunked, mmap base) · IdMap · tombstones          │
 │              binary I/O · CRC-32C · format · atomic files · MappedFile      │
 │ simd/        CPU features · kernel table (scalar | AVX2, runtime dispatch)  │
 │ concurrency/ ThreadPool · FairSharedMutex · StripedMutex                    │
 │ util/        dataset I/O · synthetic data · recall · timer · logging        │
 │ core/        Status/Result · types · config · validation · RNG · alignment  │
 └─────────────────────────────────────────────────────────────────────────────┘
```

A layer depends only on layers below it. Nothing below `collection/` knows about HTTP, JSON,
Python or the catalog directory layout. The core library `vectorforge` has no third-party
dependency; the server adds cpp-httplib and nlohmann/json, the CLI CLI11, the Python module
pybind11.

## Targets

| CMake target | Contents | Page |
|---|---|---|
| `vectorforge` (static, installed as `vectorforge::vectorforge`) | everything under `src/` except `server/` | this page |
| `vf_simd_avx2` (object library inside `vectorforge`) | the only translation unit compiled with AVX2/FMA flags | [simd.md](simd.md) |
| `vf_server` (`vectorforge::server`) | routes, JSON codec, limits, metrics | [http-api.md](http-api.md) |
| `vectorforge_cli` (`vectorforge`) | `gen-data`, `ground-truth`, `build`, `search`, `info`, `verify`, `serve` | README |
| `_vectorforge` (Python extension) | `vf.Index` | [python-api.md](python-api.md) |
| `vf_*_tests`, fuzz replays | GoogleTest suites and corpus replays | [testing.md](testing.md) |
| `vf_bench`, `vf_micro_bench`, `vf_http_bench` | benchmarks | [benchmarking.md](benchmarking.md) |

## Data path

**Insert** (`Collection::add_batch`): validate the whole batch (dimension, finiteness, duplicate
and existing ids) → normalise (cosine) on the pool → *grow* under the exclusive lock (append rows
to `VectorStore`, allocate graph nodes, publish ids in `IdMap`, tombstone replaced rows) → *link*
under the shared lock (HNSW neighbour search and back-links, in parallel), while searches continue
([concurrency.md](concurrency.md)).

**Search** (`Collection::search*`): shared lock → validate the query → backend search with a
leased `SearchContext` (HNSW: greedy descent through the upper levels, beam search on level 0,
tombstoned nodes skipped in results; Flat: blocked scan with a bounded heap) → map internal to
external ids ([hnsw.md](hnsw.md)).

**Persistence**: `save` writes a checksummed `.vfidx` through a temporary file and an atomic
rename; `load` validates the file as untrusted input and serves vectors from a read-only mapping
([storage-format.md](storage-format.md)). The catalog keeps generation files per collection and
switches `MANIFEST` atomically.

## Ownership

| Object | Owner | Notes |
|---|---|---|
| `Catalog` | application | hands out `std::shared_ptr<Collection>`; a dropped collection's files are deleted when the last reference goes |
| `Collection` | caller / catalog | pimpl; holds `std::shared_ptr<CollectionState>`, replaced by compaction |
| `CollectionState` | `Collection` | vectors, ids, tombstones, backend; nothing inside is freed while it lives |
| `VectorStore` chunks | `VectorStore` | stable addresses; mapped base kept alive by the mapping's owner |
| `SearchContext` | `ContextPool` in the backend | leased per query |
| `ThreadPool` | caller | APIs take a nullable `ThreadPool*` |

No raw owning pointers; non-owning parameters are references, pointers, spans and string views.

## Thread safety at a glance

`Collection`, `Catalog`, `Server` (after `start`) and `ThreadPool` are safe to use from any number
of threads; everything else is a value or internal. Details and the race-freedom argument:
[concurrency.md](concurrency.md).

## Error handling

Expected failures are `vf::Status` / `vf::Result<T>` values with a stable code
(`include/vectorforge/status.hpp`); `std::bad_alloc` propagates; programming errors abort through
`VF_CHECK`. The HTTP server maps codes to status codes and JSON bodies, the Python module to
exception classes. See [ADR-0001](adr/0001-status-result-errors.md).
