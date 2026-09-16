# Concurrency

This page states what VectorForge guarantees when it is used from several threads, how those
guarantees are implemented, and what was measured. Design rationale: [DESIGN.md §11](DESIGN.md#11-concurrency-design).

Source: `include/vectorforge/thread_pool.hpp`, `src/concurrency/thread_pool.cpp`,
`src/collection/collection.cpp`, `src/search/context_pool.hpp`.

## Thread-safety contract

| Type | Contract |
|---|---|
| `vf::Collection` | Thread-safe. Every member function may be called concurrently from any number of threads. |
| `vf::ThreadPool` | Thread-safe, except that the destructor (and `shutdown()`) must not race with other calls or run on one of the pool's own workers. |
| `vf::Status`, `vf::Result`, config structs | Values; no shared state. |
| `vf::simd_status()`, `vf::active_simd_level()`, `vf::distance()`, `vf::normalize()` | Thread-safe (the kernel tier is resolved once with thread-safe static initialisation). |

Observable semantics of a `Collection` shared between threads:

- **Searches** (`search`, `search_into`, `search_batch`) and other reads (`get`, `contains`,
  `size`, `stats`, `save`) run in parallel with each other.
- **Mutations** (`add`, `add_batch`, `remove`) are serialised with each other. While one runs, reads
  wait; `add_batch` holds the collection for at most about 2 ms at a time, so searches interleave
  with long ingestions and may observe a partially inserted batch. Other writers wait for the whole batch, so
  the batch's up-front validation (for example "id already exists") stays valid.
- **Removal:** a search that runs concurrently with `remove(id)` may or may not return `id`; a
  search that starts after `remove(id)` has returned never returns it.
- **`compact()`** blocks writers for its whole duration, lets searches continue while it rebuilds,
  and blocks them only for the final pointer swap. Searches before the swap use the old state,
  searches after it the new one.
- **`save()`** takes a consistent snapshot: writers wait until the file is written.
- `config()` never blocks (the configuration is immutable).

## Level A: collection-level reader/writer lock

```text
Collection::Impl {
  const CollectionConfig config;
  mutable FairSharedMutex rw;              // shared: reads; exclusive: mutations, state swap
  std::mutex writers;                      // serialises add/add_batch/remove/compact
  std::shared_ptr<CollectionState> state;  // replaced only under exclusive rw
}
```

| Operation | Locks |
|---|---|
| search, search_into, search_batch, get, contains, size, stats, save | `rw` shared |
| add, remove | `writers`, then `rw` exclusive |
| add_batch | `writers` for the whole call; `rw` exclusive per time slice of about 2 ms (at least one row per slice) |
| compact | `writers` for the whole call; `rw` shared while building the new state; `rw` exclusive for the swap |

**Why it is race-free.** Backends are thread-compatible: their const member functions only read
immutable data plus a `SearchContext` leased from a mutex-protected pool (`ContextPool`).
Every access to a `CollectionState` happens either under a shared lock with no concurrent writer,
or under the exclusive lock. `compact()` reads the old state under a shared lock (so no writer can
modify it: writers need the exclusive lock, and they are also blocked on `writers`), builds a
private new state, and swaps the pointer under the exclusive lock. The old state is destroyed after
the lock is released, when the `shared_ptr` is dropped; no reader can still hold it because readers
access the state only while holding `rw`.

**Lock ordering.** `writers` is always acquired before `rw`, and no code path acquires `writers`
while holding `rw`, so there is no lock-order cycle. `compact()` never upgrades a shared lock: it
releases the shared lock before taking the exclusive one, and `writers` guarantees that no mutation
happens in between.

**Parallel batches.** `search_batch(..., pool)` runs queries on the pool's workers while the
calling thread holds the shared lock; `ThreadPool::parallel_for` returns only after every chunk has
finished, so no worker touches the state after the lock is released. `add_batch(..., pool)`
normalises rows in parallel before taking any lock.

**Known costs.**
- Writers block searches for up to one slice (2 ms, plus the row that crosses the deadline — an HNSW
  insertion at d = 128 takes about 0.25 ms on the development machine).
- Ingestion throughput drops while searches run, because the writer yields the lock after every
  slice (measured below).
- Every search locks the fairness mutex twice (acquire and release): cache-line ping-pong across
  threads, visible only for very short queries.

## Fairness

`std::shared_mutex` makes no fairness promise. On Windows (SRWLOCK) it was measured to fail in both
directions during the `ingest` scenario (HNSW, 100K × 128, one reader, one `add_batch` writer):

| Variant | Writer section | Reader p50 | Reader p99 | Reader p99.9 | Reader max | Ingest |
|---|---|---|---|---|---|---|
| `std::shared_mutex`, `yield()` between sections | 64 rows | 0.091 ms | 0.148 ms | 225 ms | 676 ms | 4 337 vec/s |
| `std::shared_mutex` + reader hand-off counters | 64 rows | 0.086 ms | 0.138 ms | 11.2 ms | 25.0 ms | 680 vec/s |
| `FairSharedMutex` | 64 rows | 13.4 ms | 16.2 ms | 17.1 ms | 17.1 ms | 4 674 vec/s |
| `FairSharedMutex` (current) | 2 ms | 2.27 ms | 2.64 ms | 2.88 ms | 3.67 ms | 3 806 vec/s |

Idle reader for comparison: p50 0.065 ms, p99 0.10 ms. The plain `std::shared_mutex` variant (no
`yield()`) behaved like the first row (p99.9 225 ms, max 532 ms); its result file was overwritten.
All four files are in `benchmarks/results/2026-09-16_ryzen7-4800h_msvc-release_phase6a/ablation/`.

The writer re-acquired the lock in a tight loop and starved the reader (`yield()` did not help);
hand-off counters that made the writer wait for a queued reader starved the writer instead.
`detail::FairSharedMutex` (`src/concurrency/fair_shared_mutex.hpp`) is a mutex plus two condition
variables with two rules:

- **Writer preference:** while a writer waits, newly arriving readers wait, so a writer waits at most
  for the readers already inside.
- **Reader quota:** when a writer unlocks, every reader waiting at that moment is admitted before the
  next writer (`reader_quota_`), so a reader waits at most for one writer section.

Bounding the waiting time is not enough; the section length matters too. With 64-row sections a
reader waits up to one 64-row HNSW insertion (~13 ms). Sections are therefore bounded by time
(`kWriteSliceTime = 2 ms`), which fixes the reader's worst case independently of dimension and index
type, at the cost of more lock hand-offs for the writer.

## Thread pool

- Fixed number of `std::jthread` workers, one mutex-protected deque of type-erased move-only tasks
  (`detail::UniqueFunction`) and a condition variable.
- `submit(f)` / `try_submit(f)` return a `std::future`; after `shutdown()` `try_submit` returns
  `Unavailable` and `submit` returns a future holding `std::runtime_error`.
- `parallel_for(begin, end, grain, body)` splits the range into chunks of `grain` indices. At most
  `size()` helper tasks and the calling thread claim chunks from a shared atomic counter. The caller
  waits until every helper has finished; helpers keep the shared bookkeeping alive through a
  `shared_ptr`, so nothing on the caller's stack is touched after it returns. The first exception is
  rethrown in the caller; chunks not yet started are skipped. If a helper cannot be queued (pool shut
  down, allocation failure) the caller runs the remaining chunks itself.
- Nested `parallel_for` from any pool worker runs inline (a thread-local marks worker threads),
  which avoids deadlock and oversubscription.
- `shutdown()` stops accepting tasks, lets workers drain the queue, and joins them. It is idempotent
  and serialised by its own mutex; calling it from one of the pool's workers is a fatal error.

## Tests

| Test | What it checks |
|---|---|
| `tests/concurrency/test_thread_pool.cpp` | results and exceptions through futures, exact range coverage for many sizes and grains, exception propagation after all started chunks finished, caller participation, nested `parallel_for` inline, 10⁵ tiny tasks, shutdown with 500 queued tasks, concurrent clients |
| `tests/concurrency/test_concurrent_reads.cpp` | concurrent const calls return exactly the single-threaded results |
| `tests/concurrency/test_concurrent_search.cpp` | 4 readers (search, search_into, parallel search_batch) against 2 writers (add, parallel add_batch, remove), Flat and HNSW: results sorted, no duplicates, only known ids, no id removed before the query started; final counts; searches during `compact()` |
| `tests/concurrency/test_rw_stress.cpp` (label `stress`) | writer, compactor, saver and 3 readers at once; the final collection equals a model of the writer's operations; the last snapshot loads |
| `tests/unit/test_compact.cpp` | compaction keeps ids and bit-identical vectors, drops tombstones, keeps HNSW invariants, releases a file mapping; parallel batch results equal serial ones |

All of them run under ThreadSanitizer in CI (`linux-clang-tsan`). The nightly workflow
(`.github/workflows/nightly.yml`) repeats the `stress` label 100 times under TSan.

## Measurements

Development laptop (Ryzen 7 4800H, 8 cores / 16 threads, MSVC release, AVX2), 100 000 × 128-d
Gaussian-mixture vectors, k = 10. Full tables, protocol and interpretation:
[benchmarks/results/2026-09-16_ryzen7-4800h_msvc-release_phase6a](../benchmarks/results/2026-09-16_ryzen7-4800h_msvc-release_phase6a/README.md).

| `search_batch` threads | 1 | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| HNSW QPS (ef = 64) | 9 667 | 17 779 | 30 689 | 39 532 | 41 551 | 46 282 |
| Flat QPS | 381 | 706 | 1 073 | 1 721 | 2 117 | 1 187 |

- HNSW reaches 4.8× at 16 threads; Flat reaches 5.5× at 12 threads and drops at 16 (reproduced;
  most likely memory-bound scans competing for per-core cache — not profiled). Use at most one
  thread per physical core for Flat batches.
- Ingest (one `add_batch` writer, one reader, HNSW): the reader's median latency rises from 0.065 ms
  to 2.27 ms (max 3.67 ms) while 3 806 vectors/s are inserted. Flat 10⁶: 12.4 ms → 20.9 ms median
  at 123 658 vectors/s. Level A trades search latency during ingestion for simplicity; Level B
  (Phase 6b) is the fix.
