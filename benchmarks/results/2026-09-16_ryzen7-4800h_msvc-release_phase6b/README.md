# Phase 6b concurrent HNSW insertion — 2026-09-16

Raw data:

| File | Content |
|---|---|
| [`build_hnsw_100k_d128.json`](build_hnsw_100k_d128.json) | `vf_bench --scenario build`: 100 000 × 128-d through `add_batch` (batches of 10 000), Coarse serial and Concurrent with 1–16 threads; validator result and recall@10 at ef 16–128 per graph |
| [`build_hnsw_100k_d768.json`](build_hnsw_100k_d768.json) | the same at d = 768 with 1, 4, 8, 16 threads |
| [`ingest_hnsw_coarse.json`](ingest_hnsw_coarse.json) | `vf_bench --scenario ingest`, Level A (Coarse) |
| [`ingest_hnsw_concurrent_w1.json`](ingest_hnsw_concurrent_w1.json) | the same, Level B with a single-threaded writer |
| [`ingest_hnsw_concurrent_w4.json`](ingest_hnsw_concurrent_w4.json) | the same, Level B with a 4-thread writer |

## Environment

Same machine and toolchain as
[Phase 6a](../2026-09-16_ryzen7-4800h_msvc-release_phase6a/README.md#environment): AMD Ryzen 7
4800H (8 cores / 16 threads), 15.4 GB DDR4, Windows 11, MSVC 19.50 `msvc-release`, AVX2 tier.
Source: `030bb220c956-dirty`, the working tree that became the Phase 6b commit. Single process run
per file; no other benchmark ran at the same time. After these runs, `add_batch` was changed to hold
its `link` mutex from the grow section to the end of the link section (a fix for saves, see
ADR-0003); only `save()` contends for that mutex, and these scenarios do not save.

**Protocol.** Seeded Gaussian mixture (seed 1, 100 clusters, σ = 0.1), L2, M = 16,
ef_construction = 200, k = 10; queries from seed 2 (1 000 at d = 128, 500 at d = 768). Build time is
wall time of the `add_batch` calls only (ground truth and recall are computed afterwards with a
pool). `ingest`: 50 000 vectors preloaded, one reader measures 1 000 idle queries, then queries in
a loop while the writer inserts the other 50 000 in batches of 1 000 (ef_search = 64).

## Build scaling

| Threads | d = 128 build (s) | speedup | d = 768 build (s) | speedup |
|---|---|---|---|---|
| 1 (Coarse) | 18.27 | 1.00 | 40.42 | 1.00 |
| 1 (Concurrent) | 17.48 | 1.05 | 40.19 | 1.01 |
| 2 | 9.02 | 2.03 | — | — |
| 4 | 5.07 | 3.60 | 14.62 | 2.76 |
| 6 | 3.73 | 4.90 | — | — |
| 8 | 2.96 | 6.18 | 11.64 | 3.47 |
| 12 | 2.96 | 6.17 | — | — |
| 16 | 2.56 | 7.13 | 12.21 | 3.31 |

Recall@10 of every graph (d = 128): ef 16 → 0.8167–0.8168, ef 32 → 0.9382–0.9383, ef 64 →
0.9906–0.9907, ef 128 → 0.9998. d = 768: ef 32 → 0.8866–0.8868, ef 64 → 0.9698, ef 128 → 0.9926.
Every graph passed `HnswValidator::check_invariants`; no unrepaired orphans. Level-0 unreachable
nodes: 0 at d = 128; 1 at d = 768 in *every* build including the serial one (a property of this
data set, not of parallel insertion).

**Interpretation.**
- Parallel insertion costs no measurable recall here: the largest difference to the serial build is
  0.0002 (one hit in 10 000 or 5 000).
- d = 128 scales to 6.2× on the 8 physical cores; SMT adds 15% at 16 threads (12 threads measured
  the same as 8, within run-to-run variation of a single run). This is better than batch *search*
  scaling on the same machine (4.1× at 8 threads, Phase 6a). Why is not established (no profiler
  was run); one candidate is that neighbour selection re-reads rows the insertion already loaded.
- d = 768 stops at 3.5×. Each insertion reads hundreds of 3 KB rows (ef_construction = 200
  candidates and their neighbours), so the build is most likely memory-bound, the same pattern as
  Flat search in Phase 6a (not profiled). Beyond 8 threads it gets slightly slower.
- Lock contention does not appear to limit scaling at d = 128 (6.2× on 8 cores), so the per-node
  spinlock alternative in DESIGN §11.4 was **not run**; there is no contention signal to act on. No
  profiler was used.
- The serial Concurrent build is identical to the Coarse build (tested byte for byte in
  `test_parallel_build`), and 4% faster here — within single-run variation.

## Search latency during ingestion

| Mode | Writer threads | Ingest rate | Reader p50 | p99 | p99.9 | max | Reader queries |
|---|---|---|---|---|---|---|---|
| idle (no writer) | — | — | 0.065 ms | 0.092 ms | 0.144 ms | — | 1 000 |
| Coarse (Level A) | 1 | 3 842 vec/s | 2.27 ms | 2.61 ms | 2.85 ms | 3.92 ms | 5 775 |
| Concurrent (Level B) | 1 | 4 567 vec/s | 0.084 ms | 0.148 ms | 0.327 ms | 2.39 ms | 128 425 |
| Concurrent (Level B) | 4 | 15 293 vec/s | 0.098 ms | 0.202 ms | 0.402 ms | 2.77 ms | 32 229 |

**Interpretation.**
- Level B removes the Level A wait: the reader's median during ingestion is 1.3× its idle median
  instead of 35×, and it completed 22× more queries in a shorter window.
- The writer is also faster (+19% with one thread) because it no longer yields the exclusive lock
  to the reader between rows; with four threads it inserts 4× faster than Level A.
- The remaining tail (max 2.4–2.8 ms) is the grow section (appending up to 64 × threads rows and
  allocating nodes, bounded at 2 ms) plus the write-preferring lock hand-off.
- With 4 writer threads the reader competes for CPU with 4 linkers and their cache traffic; its
  median rises from 0.084 to 0.098 ms.

**Decision (DESIGN §21 Phase 6b gate):** TSan clean (CI preset `linux-clang-tsan`), validator passes
after every parallel build, recall within 0.0002 of serial, measured speedups above.
`Concurrency::Concurrent` is the default; `Coarse` stays available.
[ADR-0003](../../../docs/adr/0003-concurrent-hnsw-insert.md).

## Reproduce

```powershell
. .\tools\dev-env.ps1
cmake --preset msvc-release; cmake --build --preset msvc-release
$b = "out\build\msvc-release\benchmarks\vf_bench.exe"
& $b --scenario build --n 100000 --dim 128 --queries 1000 --ef-search 16,32,64,128 --threads 1,2,4,6,8,12,16 --batch 10000 --out build_hnsw_100k_d128.json
& $b --scenario build --n 100000 --dim 768 --queries 500 --ef-search 32,64,128 --threads 1,4,8,16 --batch 10000 --out build_hnsw_100k_d768.json
& $b --scenario ingest --n 100000 --dim 128 --index hnsw --queries 1000 --ef-search 64 --batch 1000 --concurrency coarse --out ingest_hnsw_coarse.json
& $b --scenario ingest --n 100000 --dim 128 --index hnsw --queries 1000 --ef-search 64 --batch 1000 --concurrency concurrent --out ingest_hnsw_concurrent_w1.json
& $b --scenario ingest --n 100000 --dim 128 --index hnsw --queries 1000 --ef-search 64 --batch 1000 --concurrency concurrent --writer-threads 4 --out ingest_hnsw_concurrent_w4.json
```
