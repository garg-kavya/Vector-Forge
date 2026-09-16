# ADR-0003: Concurrent HNSW insertion with a grow/link split

- **Status:** Accepted
- **Date:** 2026-09-16
- **Related:** docs/DESIGN.md §11.4, §11.7, §21 (Phase 6b); docs/concurrency.md "Level B"

## Context

Level A (Phase 6a) serialises every insertion with every search. During ingestion a search waits
for the current 2 ms write section, so HNSW queries that take 0.065 ms idle take 2.27 ms
(`benchmarks/results/2026-09-16_ryzen7-4800h_msvc-release_phase6a`), and index builds use one
core. DESIGN §11.4 planned lock-free readers of atomic link lists, striped writer locks, and a
fixed-capacity atomic chunk directory so that vectors and nodes could also be appended while
searches run. Requirements: no data race (TSan clean), graphs that pass the validator, recall close
to a serial build, an unchanged failure contract where possible, and a fallback.

## Decision

1. **Split insertion into grow and link.** Everything that changes directories or can fail —
   vector append, graph node allocation, id map and tombstone updates, sizing of insert scratch —
   runs in a short exclusive section (at most `64 × threads` rows and 2 ms). Linking the grown rows
   (neighbour search, own lists, back-links) runs under the shared lock, in parallel on the
   caller's `ThreadPool`, while searches continue. Chunk directories therefore never change while
   anyone reads them, and no atomic directory or id-reservation (`PENDING`) state is needed.
2. **Atomic link lists.** List counts and slots are `std::atomic<uint32_t>`; readers load them
   without locks. Writers of a node's lists hold one of 4 096 striped mutexes, one at a time.
3. **Entry point.** Packed `(id, level)` in one `std::atomic<uint64_t>`. An insertion above the top
   level it observed holds `top_mutex_` for its whole duration and re-reads the entry point, so
   concurrent high-level insertions are serialised (as in hnswlib).
4. **Saves** hold a `link` mutex, which each grow-and-link section holds from growing to linking
   (lock order `writers` → `link` → `rw`), so a snapshot never contains allocated but unlinked
   nodes. (The first version took `link` only for linking; the stress test caught a snapshot taken
   between the two sections, whose unlinked top-level node failed the graph validator.)
5. **Mode switch.** `CollectionConfig::concurrency` / `LoadOptions::concurrency`:
   `Concurrent` (default) or `Coarse` (Level A). The setting is not stored in index files.
6. **Failure contract.** A failure in the grow section keeps Level A's per-row strong guarantee.
   The only possible failure while linking is `std::bad_alloc` from the neighbour search's candidate
   queue, before the node is reachable; such a row is tombstoned, its id restored to its previous
   mapping (`IdMap::revert`), and the exception propagates.

## Alternatives considered

| Alternative | Why not chosen |
|---|---|
| Fixed-capacity atomic chunk directories and per-row appends under the shared lock (DESIGN §11.4 as planned) | More shared state to argue about (directory pointers, id allocation, `PENDING` ids, atomic tombstones, a separate writer gate for saves). Grow sections cost microseconds per row, and the measured reader tail with them is 2.4–2.8 ms max / 0.33 ms p99.9. |
| Per-node 1-byte spinlocks instead of striped mutexes | Planned only if striping showed contention; d = 128 builds scale 6.2× on 8 cores, so there was no signal. Not run. |
| CAS loop on the entry point without a mutex | A concurrent insertion above the old top level could link on fewer levels than it joins, leaving upper-level nodes unreachable; the mutex is taken only for rare top-level insertions. |
| Keep inserts exclusive and only parallelise index builds offline | Leaves search latency during ingestion at Level A values, which is the main user-visible problem. |
| `Concurrent` opt-in instead of default | All Phase 6b gates passed; the serial Concurrent build equals the Coarse build byte for byte, so the default changes nothing for single-threaded users except shorter search waits. |

## Consequences

- Searches during ingestion: p50 0.084 ms instead of 2.27 ms; builds 6.2× faster on 8 cores at
  d = 128 (3.5× at d = 768, memory-bound).
- Parallel builds are not bit-identical to serial builds; determinism tests use serial builds.
- A search concurrent with a link section may miss the newest rows; `add()` stays exclusive.
- A rare `bad_alloc` while linking leaves a tombstoned row (counted in `deleted_count` until
  `compact()`), which is weaker than Level A's "no trace".
- Every list write is an atomic store and every back-link update takes an uncontended mutex; the
  serial Concurrent build was not slower than Coarse (17.5 s vs 18.3 s, single runs).
- `HnswGraph` is no longer trivially movable (explicit move operations) and `LinkView` no longer
  exposes a span.

## Validation

- `tests/concurrency/test_parallel_build.cpp`: serial Concurrent == Coarse (canonical bytes);
  1–8 thread builds pass the validator, full level-0 reachability, recall within 0.02 of serial;
  upserts; empty-graph entry races.
- `tests/concurrency/test_concurrent_insert_search.cpp` (label `stress`, nightly ×100 under TSan):
  readers, a remover and a saver during parallel ingestion.
- `tests/alloc/test_exception_safety.cpp`: allocation failure injection in both modes.
- `benchmarks/results/2026-09-16_ryzen7-4800h_msvc-release_phase6b`: build scaling (recall per
  graph within 0.0002 of serial) and ingest latency, Coarse vs Concurrent.
