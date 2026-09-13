# Phase 3 HNSW baselines — 2026-09-14

Raw data (host name and local executable path redacted from the Google Benchmark files):

| File | Content |
|---|---|
| [`hnsw_l2_100k_d128.json`](hnsw_l2_100k_d128.json) | 100K × 128-d, L2: build, ef_search sweep, Flat latency |
| [`hnsw_cosine_100k_d128.json`](hnsw_cosine_100k_d128.json) | same, cosine |
| [`hnsw_l2_100k_d128_simple.json`](hnsw_l2_100k_d128_simple.json) | same as L2 with `select = simple` (ablation) |
| [`hnsw_l2_10k_d128.json`](hnsw_l2_10k_d128.json) | 10K × 128-d, L2 |
| [`hnsw_ip_100k_d32.json`](hnsw_ip_100k_d32.json), [`hnsw_ip_10k_d16.json`](hnsw_ip_10k_d16.json) | inner product: reachability and recall |
| [`visited.json`](visited.json) | visited-set strategies (final code) |
| [`visited_before_branch_free.json`](visited_before_branch_free.json) | same, before `VisitedSet::visit` was made branch-free |

## Environment

| Item | Value |
|---|---|
| CPU | AMD Ryzen 7 4800H (Zen 2), 8 cores / 16 threads, AVX2+FMA, no AVX-512 |
| OS | Windows 11 Home 10.0.26200 |
| Power | AC power, battery 100%, custom power plan (laptop; thermal throttling not controlled) |
| Compiler | MSVC 19.50.35728, `msvc-release` preset (`/O2`, no `/arch`, no LTO) |
| SIMD tier | scalar |
| Threads | 1 (build and queries) |
| Source | `vf_git_sha` = `fa54bc14455d-dirty`: the working tree that became the Phase 3 commit (parent `fa54bc1` plus the Phase 3 changes); only formatting-neutral documentation changed after the runs |

**Protocol (minimal, Phase 3):** one process per configuration, **one run each** (no repetitions),
1 000 queries (200/500 for the IP runs), one unrecorded warm-up pass per `ef_search`, per-query
`steady_clock` latency, nearest-rank percentiles. With 1 000 samples P99 rests on 10 samples. These are
baselines for later comparison on this machine, not performance claims; the full protocol (≥ 5 runs,
10K queries) arrives with the Phase 9 suite.

**Data:** `vectorforge`'s seeded Gaussian mixture (100 clusters, centres uniform in [−1, 1)^d,
σ = 0.1 per component; queries from the same mixture, different seed), except the 10K/16-d IP run
(50 clusters, σ = 0.25). Ground truth: Flat collection in the same process. Index: `M = 16`,
`ef_construction = 200`, heuristic selection with `keep_pruned` and orphan repair, `k = 10`.

## Reproduce

```powershell
. .\tools\dev-env.ps1
cmake --preset msvc-release
cmake --build --preset msvc-release
$b = "out\build\msvc-release\benchmarks\vf_bench.exe"
& $b --n 100000 --queries 1000 --dim 128 --metric l2 --out hnsw_l2_100k_d128.json
& $b --n 100000 --queries 1000 --dim 128 --metric cosine --out hnsw_cosine_100k_d128.json
& $b --n 100000 --queries 1000 --dim 128 --metric l2 --selection simple --skip-flat-timing --out hnsw_l2_100k_d128_simple.json
& $b --n 10000 --queries 1000 --dim 128 --metric l2 --out hnsw_l2_10k_d128.json
& $b --n 100000 --queries 200 --dim 32 --metric ip --ef-search 32,128 --skip-flat-timing --out hnsw_ip_100k_d32.json
& $b --n 10000 --queries 500 --dim 16 --clusters 50 --spread 0.25 --metric ip --ef-search 16,32,64 --skip-flat-timing --out hnsw_ip_10k_d16.json
out\build\msvc-release\benchmarks\vf_micro_bench.exe --benchmark_filter=BenchVisited --benchmark_repetitions=3 `
  --benchmark_min_time=0.2s --benchmark_report_aggregates_only=true --benchmark_out=visited.json --benchmark_out_format=json
```

## Recall vs ef_search — 100K × 128-d, L2

Build: 40.9 s, 322.7 M distance computations, 18.8 MiB index structures, invariants OK, 0 unreachable
nodes. Flat on the same queries: 7.63 ms mean latency, 131 QPS.

| ef_search | recall@10 | min | QPS | p50 µs | p95 µs | p99 µs | distances / query |
|---|---|---|---|---|---|---|---|
| 10 | 0.7084 | 0.0 | 13 996 | 70.2 | 89.5 | 101.9 | 387.6 |
| 16 | 0.8167 | 0.4 | 11 020 | 89.6 | 107.4 | 126.5 | 477.6 |
| 32 | 0.9383 | 0.6 | 7 528 | 132.1 | 150.7 | 174.3 | 660.4 |
| 64 | 0.9906 | 0.8 | 5 358 | 185.2 | 210.1 | 243.8 | 873.8 |
| 128 | 0.9998 | 0.9 | 4 047 | 242.3 | 282.9 | 330.3 | 1 039.0 |
| 256 | 1.0000 | 1.0 | 3 325 | 299.0 | 331.0 | 353.8 | 1 111.1 |
| 512 | 1.0000 | 1.0 | 2 629 | 378.7 | 418.0 | 451.6 | 1 131.8 |

## Recall vs ef_search — 100K × 128-d, cosine

Build: 38.8 s, invariants OK, 0 unreachable nodes. Flat: 7.40 ms mean latency, 135 QPS.

| ef_search | recall@10 | min | QPS | p50 µs | p99 µs | distances / query |
|---|---|---|---|---|---|---|
| 10 | 0.7044 | 0.0 | 15 120 | 64.9 | 97.7 | 383.3 |
| 16 | 0.8221 | 0.3 | 11 694 | 84.0 | 118.6 | 475.3 |
| 32 | 0.9411 | 0.5 | 7 994 | 123.3 | 164.6 | 658.8 |
| 64 | 0.9903 | 0.8 | 5 562 | 176.8 | 255.2 | 872.6 |
| 128 | 0.9995 | 0.9 | 4 293 | 229.4 | 288.5 | 1 038.1 |
| 256 | 1.0000 | 1.0 | 3 183 | 300.7 | 491.1 | 1 111.2 |
| 512 | 1.0000 | 1.0 | 2 666 | 371.9 | 452.5 | 1 135.5 |

## 10K × 128-d, L2

Build 2.76 s, 0 unreachable. Flat: 0.771 ms, 1 297 QPS. Recall@10: 0.9905 (ef 10), 0.9985 (16), 0.9999 (32),
1.0 from ef 64; QPS 41 402 / 37 479 / 30 373 / 22 765 (ef 10/16/32/64).

## Neighbour selection ablation — 100K × 128-d, L2

| Selection | build s | build distances | unreachable (level 0) | orphans unrepaired | recall@10 ef 64 | ef 512 |
|---|---|---|---|---|---|---|
| heuristic + keep_pruned | 40.9 | 322.7 M | 0 | 0 | 0.9906 | 1.0000 |
| simple (M closest) | 20.9 | 97.3 M | 112 | 1 548 | 0.7838 | 0.8601 |

## Inner product

| Run | unreachable (level 0) | orphan repairs | recall@10 |
|---|---|---|---|
| 10K × 16-d (50 clusters, σ 0.25) | 118 (1.2 %) | 1 | 0.9990 / 0.9998 / 0.9998 at ef 16 / 32 / 64 |
| 100K × 32-d | 38 850 (38.9 %) | 13 338 (708 unrepaired) | 0.9950 / 1.0000 at ef 32 / 128 |

## Visited set (median per simulated search: reset + `visits` visits, about half repeats)

| n | visits | epoch array (before) | epoch array (branch-free) | clear byte array | `std::unordered_set` |
|---|---|---|---|---|---|
| 10 000 | 512 | 1 700 ns | 529 ns | 459 ns | 16 410 ns |
| 10 000 | 4 096 | 18 160 ns | 5 642 ns | 3 104 ns | 124 517 ns |
| 1 000 000 | 512 | 1 919 ns | 880 ns | 11 411 ns | 16 121 ns |
| 1 000 000 | 4 096 | 23 666 ns | 7 614 ns | 18 563 ns | 132 122 ns |

## Observations and decisions (this machine and compiler only)

1. **HNSW vs Flat at 100K × 128-d:** at ef_search = 64 (recall@10 0.9906) single-query throughput was
   5 358 QPS versus 131 QPS for Flat on the same queries (about 41×); at ef_search = 128 (recall 0.9998)
   4 047 QPS (about 31×). Single runs, scalar kernels.
2. **Recall is monotone in ef_search** in every sweep. The 100K Gaussian mixture (σ = 0.1) is harder than
   the 10K sets: recall@10 ≥ 0.99 needs ef_search = 64 at 100K versus 10 at 10K.
3. **Keep the heuristic.** Simple selection halves build time but caps recall at 0.86 even at ef = 512 and
   leaves 112 nodes unreachable — the connectivity failure the heuristic exists to prevent (DESIGN §9.7).
4. **Distance computations dominate query cost:** latency tracks distances/query (≈ 0.2 µs per 128-d
   distance including graph overhead). SIMD kernels (Phase 5) are the next lever.
5. **Visited set:** making `visit()` branch-free sped the epoch array up 2.2–3.2× in isolation. The epoch
   array is 13× (512 visits) and 2.4× (4 096 visits) faster than clearing a byte array at n = 1M, where
   the O(n) clear dominates; at n = 10K clearing a byte array takes 13–45 % less time. The epoch array
   stays the default because the per-search cost must not grow with the collection. No end-to-end A/B of
   the branch-free change was recorded; given observation 4 its end-to-end effect is expected to be small.
6. **Inner product on unnormalised data leaves nodes unreachable** (39 % at 100K) while recall for
   in-distribution queries stays high; see docs/hnsw.md "Known limitations". L2 and cosine graphs had no
   unreachable nodes, and orphan repair never triggered for them.
