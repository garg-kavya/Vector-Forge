# Phase 6a concurrency measurements — 2026-09-16

Raw data:

| File | Content |
|---|---|
| [`threads_hnsw_100k_d128.json`](threads_hnsw_100k_d128.json) | `vf_bench --scenario threads`: `search_batch` QPS over a saved HNSW index for 1–16 threads |
| [`threads_flat_100k_d128.json`](threads_flat_100k_d128.json) | the same for a Flat index |
| [`threads_flat_100k_d128_recheck.json`](threads_flat_100k_d128_recheck.json) | second Flat run at 1, 12, 14, 16 threads (checks the 16-thread drop) |
| [`ingest_hnsw_100k_d128.json`](ingest_hnsw_100k_d128.json) | `vf_bench --scenario ingest`: one reader's search latency while one writer calls `add_batch` (HNSW) |
| [`ingest_flat_1m_d128.json`](ingest_flat_1m_d128.json) | the same for Flat with 10⁶ vectors |
| `ablation/ingest_hnsw_*.json` | the ingest scenario with earlier lock designs (see "Fairness") |

## Environment

| Item | Value |
|---|---|
| CPU | AMD Ryzen 7 4800H (Zen 2), 8 cores / 16 threads, L2 512 KiB per core, L3 4 MiB per CCX |
| RAM | 15.4 GB DDR4 (dual channel) |
| OS | Windows 11 Home 10.0.26200 |
| Power | AC power (laptop; thermal state not controlled) |
| Compiler | MSVC 19.50.35728, `msvc-release` preset, AVX2 kernel tier |
| Source | `c6c2f9dd9a7d-dirty`: the working tree that became the Phase 6a commit. The ablation files were produced by intermediate versions of `src/collection/collection.cpp` |

**Protocol.** Data: seeded Gaussian mixture (seed 1), d = 128, L2; queries from seed 2. HNSW:
M = 16, ef_construction = 200, ef_search = 64, k = 10. `threads`: the index is loaded into memory
(no mmap), each thread count gets its own pool (`threads − 1` workers; the caller participates),
one warm-up pass, then 5 passes over the query set; the median pass is reported. `ingest`: half of
the data is preloaded; the reader measures 1 000 (Flat: 200) idle queries, then runs queries in a
loop while the writer inserts the other half in batches of 1 000 (Flat: 10 000). Single process
run per file.

## Thread scaling (`search_batch`)

| Threads | HNSW QPS | speedup | Flat QPS | speedup |
|---|---|---|---|---|
| 1 | 9 667 | 1.00 | 381 | 1.00 |
| 2 | 17 779 | 1.84 | 706 | 1.85 |
| 4 | 30 689 | 3.17 | 1 073 | 2.81 |
| 6 | 37 307 | 3.86 | 1 387 | 3.64 |
| 8 | 39 532 | 4.09 | 1 721 | 4.51 |
| 12 | 41 551 | 4.30 | 2 117 | 5.55 |
| 16 | 46 282 | 4.79 | 1 187 | 3.11 |

HNSW (20 000 queries) and Flat (2 000 queries); min/max of the 5 passes are in the files.

**Interpretation.**
- Scaling is close to linear up to 4 threads and flattens beyond the 8 physical cores; SMT
  siblings add about 17% (HNSW, 8 → 16). An earlier run of the same HNSW configuration during development
  measured 10 342 → 49 804 QPS (4.82×), so single-run differences of ±7% are noise on this laptop.
- HNSW scaling stops well below 8× at 8 threads. Candidate causes, **not verified with a profiler**:
  memory bandwidth (every query touches roughly 0.5–1 MB of cold vector and graph data), shared
  L3 (4 MiB per 4-core CCX), and laptop frequency scaling when all cores are busy.
- Flat falls from 2 117 QPS at 12 threads to 1 187 at 16 and was reproduced (2 051 → 1 545 → 1 125
  at 12/14/16). A Flat query scans 51 MB; at 12 threads the machine already moves ~100 GB/s of
  vector data, which is above what dual-channel DDR4 delivers sequentially, so the scan relies on
  the blocked kernel keeping blocks cache-hot. With 16 threads two scans share each core's L2 and
  the blocks no longer fit. This explanation is a hypothesis; the practical consequence is that
  Flat batch search should not use more threads than physical cores on this machine.
- The per-query cost of the fair lock (two mutex acquisitions) is invisible at these query times.

## Search latency during ingestion (one reader, one writer)

| Index | Ingest rate | Idle p50 | Idle p99 | Busy p50 | Busy p99 | Busy p99.9 | Busy max |
|---|---|---|---|---|---|---|---|
| HNSW 100K | 3 806 vec/s | 0.065 ms | 0.101 ms | 2.27 ms | 2.64 ms | 2.88 ms | 3.67 ms |
| Flat 1M | 123 658 vec/s | 12.4 ms | 13.9 ms | 20.9 ms | 34.6 ms | 43.6 ms | 43.6 ms |

**Interpretation.** Level A serialises readers and writers, so while a writer is busy each search
waits for the current write section (at most ~2 ms plus one row) and the writer waits for the
search. For HNSW the query itself (0.065 ms) is small against the 2 ms section, so busy latency is
dominated by the section length: bounded and predictable, but 35× the idle median. For Flat the
query (12 ms) dominates and the added wait is one section plus the writer's time between sections.
Removing this wait needs Level B (Phase 6b: searches that do not block on inserts).

## Fairness (ablation, HNSW ingest)

| Lock | Section | Busy p50 | Busy p99.9 | Busy max | Ingest |
|---|---|---|---|---|---|
| `std::shared_mutex` + `yield()` | 64 rows | 0.091 ms | 225 ms | 676 ms | 4 337 vec/s |
| `std::shared_mutex` + reader hand-off | 64 rows | 0.086 ms | 11.2 ms | 25.0 ms | 680 vec/s |
| `FairSharedMutex` | 64 rows | 13.4 ms | 17.1 ms | 17.1 ms | 4 674 vec/s |
| `FairSharedMutex` (shipped) | 2 ms | 2.27 ms | 2.88 ms | 3.67 ms | 3 806 vec/s |

With Windows SRWLOCK the writer re-acquired the lock ahead of the waiting reader for up to 0.7 s.
Making the writer wait for queued readers fixed the reader but starved the writer (−84% ingest).
The fair lock bounds both waits; the time-bounded section then trades 19% ingest throughput (against
64-row sections) for a 5× lower reader tail. Details: [docs/concurrency.md](../../../docs/concurrency.md#fairness).

## Reproduce

```powershell
. .\tools\dev-env.ps1
cmake --preset msvc-release; cmake --build --preset msvc-release
$b = "out\build\msvc-release\benchmarks\vf_bench.exe"
& $b --scenario save --n 100000 --dim 128 --index flat --index-file flat_d128.vfidx
& $b --scenario save --n 100000 --dim 128 --index hnsw --index-file hnsw_d128.vfidx
& $b --scenario threads --index-file hnsw_d128.vfidx --queries 20000 --ef-search 64 --threads 1,2,4,6,8,12,16 --repeat 5 --out threads_hnsw_100k_d128.json
& $b --scenario threads --index-file flat_d128.vfidx --queries 2000 --threads 1,2,4,6,8,12,16 --repeat 5 --out threads_flat_100k_d128.json
& $b --scenario ingest --n 100000 --dim 128 --index hnsw --queries 1000 --ef-search 64 --batch 1000 --out ingest_hnsw_100k_d128.json
& $b --scenario ingest --n 1000000 --dim 128 --index flat --queries 200 --batch 10000 --out ingest_flat_1m_d128.json
```
