# Phase 5 SIMD measurements — 2026-09-14

Raw data (host name and executable path redacted from Google Benchmark contexts):

| File | Content |
|---|---|
| [`kernels.json`](kernels.json) | Google Benchmark: dot, l2sq, norm2, l2sq 1-to-N (1 024 and 100 000 rows) for `scalar`, `scalar_autovec`, `avx2_acc4`, `avx2_acc1`, `avx2_acc4_masked`; 5 repetitions, aggregates |
| [`dispatch.json`](dispatch.json) | Google Benchmark: table-dispatched vs direct vs inlined AVX2 distance loop; 20 repetitions, random interleaving |
| `save_hnsw_100k_d{128,768}_*.json` | `vf_bench --scenario save`: build (100 000 inserts through `Collection::add_batch`) and save, per tier |
| `sweep_hnsw_100k_d{128,768}_{scalar,avx2}_prefetch_{off,on}.json` | `vf_bench` sweep over one saved index file per dimension (same graph for every run): recall@10, QPS, latency, distance computations; Flat QPS on the same queries |
| `build_hnsw_100k_d{128,768}_avx2_prefetch_{off,on}_run{1,2}.json` | `vf_bench` sweep that builds the graph: build time with and without prefetch, two runs each |

## Environment

| Item | Value |
|---|---|
| CPU | AMD Ryzen 7 4800H (Zen 2), 8 cores / 16 threads, AVX2 + FMA, L1d 32 KiB, L2 512 KiB, L3 4 MiB per CCX |
| RAM | 15.4 GB DDR4 |
| OS | Windows 11 Home 10.0.26200 |
| Power | AC power, custom power plan (laptop; thermal state not controlled) |
| Compiler | MSVC 19.50.35728, `msvc-release` preset (`/O2`; AVX2 object `/arch:AVX2`) |
| Library | Google Benchmark 1.9.5 |
| Threads | 1 |
| Source | `vf_git_sha` = `a32559d829b4-dirty`: the working tree that became the Phase 5 commit. The sweep and build runs used `--prefetch` as an on/off flag; it is now `--prefetch on\|off` |

**Protocol and limits.** Micro benchmarks: 5 repetitions (dispatch: 20, randomly interleaved), medians
reported. Macro: one fresh process per file, 1 000 queries (200 for the build runs), k = 10, warm-up
pass before each measured pass; single run per configuration except the build ablation (two runs).
Data: seeded Gaussian mixture (100 clusters, σ = 0.1), L2, M = 16, ef_construction = 200. No profiler
was run; the distance-time shares below are estimates from micro results and distance counters.

## Reproduce

```powershell
. .\tools\dev-env.ps1
cmake --preset msvc-release; cmake --build --preset msvc-release
$m = "out\build\msvc-release\benchmarks\vf_micro_bench.exe"
& $m --benchmark_filter="Bench(Dot|L2sq|Norm2)" --benchmark_repetitions=5 --benchmark_min_time=0.2s `
  --benchmark_report_aggregates_only=true --benchmark_out=kernels.json --benchmark_out_format=json
& $m --benchmark_filter=BenchDispatch --benchmark_repetitions=20 --benchmark_enable_random_interleaving=true `
  --benchmark_min_time=0.2s --benchmark_report_aggregates_only=true --benchmark_out=dispatch.json --benchmark_out_format=json
$b = "out\build\msvc-release\benchmarks\vf_bench.exe"
& $b --scenario save --n 100000 --dim 128 --index hnsw --index-file hnsw_d128.vfidx --simd avx2 --out save_hnsw_100k_d128_avx2.json
& $b --scenario save --n 100000 --dim 128 --index hnsw --index-file scalar_built.vfidx --simd scalar --out save_hnsw_100k_d128_scalar.json
& $b --index-file hnsw_d128.vfidx --queries 1000 --simd scalar --prefetch off --out sweep_hnsw_100k_d128_scalar_prefetch_off.json
#   ... likewise --simd {scalar,avx2} x --prefetch {off,on}
& $b --scenario save --n 100000 --dim 768 --index hnsw --index-file hnsw_d768.vfidx --simd avx2 --out save_hnsw_100k_d768_avx2.json
& $b --index-file hnsw_d768.vfidx --queries 1000 --ef-search 16,32,64,128,256 --simd scalar --prefetch off --out sweep_hnsw_100k_d768_scalar_prefetch_off.json
& $b --dim 768 --queries 200 --ef-search 64 --skip-flat-timing --simd avx2 --prefetch off --out build_hnsw_100k_d768_avx2_prefetch_off_run1.json
```

## 1. Kernels (median ns per call, cache-hot, aligned / one-float offset)

`l2sq` (`dot` within a few percent; see JSON):

| dim | scalar | scalar_autovec | avx2_acc4 | avx2_acc1 | avx2_acc4_masked |
|---|---|---|---|---|---|
| 8 | 3.34 / 3.40 | 2.86 / 2.87 | 4.06 / 4.07 | 3.59 / 3.59 | 5.00 / 5.00 |
| 16 | 5.88 / 5.82 | 3.41 / 3.35 | 4.45 / 4.48 | 4.14 / 4.14 | 5.48 / 5.48 |
| 100 | 52.83 / 53.01 | 11.52 / 12.02 | 8.51 / 9.27 | 12.12 / 12.19 | 8.10 / 8.38 |
| 128 | 72.55 / 72.97 | 13.44 / 13.55 | 7.79 / 8.92 | 14.63 / 14.83 | 8.64 / 9.17 |
| 384 | 259.88 / 258.73 | 36.33 / 37.31 | 17.91 / 23.38 | 42.87 / 43.13 | 18.01 / 23.68 |
| 768 | 528.84 / 528.46 | 75.92 / 76.65 | 32.17 / 42.06 | 94.05 / 96.58 | 32.22 / 42.24 |
| 1536 | 1067.80 / 1079.81 | 145.86 / 146.03 | 60.49 / 79.32 | 206.53 / 209.93 | 60.49 / 80.77 |
| 1537 | 1073.10 / 1076.07 | 145.54 / 145.25 | 60.57 / 79.75 | 207.95 / 211.35 | 61.07 / 80.10 |

`dot`, aligned: d = 128: 69.18 / 10.67 / 6.80 / 14.19 / 7.52; d = 768: 520.02 / 69.53 / 30.46 / 93.15 /
30.48. `norm2`, d = 768: 520 / 66.3 / 26.0 / 93.9 / 30.8.

`l2sq_1_to_n` (median time for all rows):

| rows × dim | scalar | scalar_autovec | avx2_acc4 | avx2_acc1 | avx2_acc4_masked |
|---|---|---|---|---|---|
| 1 024 × 128 | 73.7 µs | 74.5 µs | 9.21 µs | 12.6 µs | 9.49 µs |
| 1 024 × 768 | 552 µs | 548 µs | 43.7 µs | 97.4 µs | 43.7 µs |
| 1 024 × 1536 | 1 140 µs | 1 140 µs | 266 µs | 289 µs | 256 µs |
| 100 000 × 128 | 7.48 ms | 7.51 ms | 2.45 ms | 2.70 ms | 2.49 ms |
| 100 000 × 768 | 54.9 ms | 54.3 ms | 14.5 ms | 14.8 ms | 14.4 ms |

## 2. Dispatch cost (median ns per distance, 4 096 random rows)

| dim, rows | Table (production) | Direct call | Inlined into AVX2 loop |
|---|---|---|---|
| 128, 1 024 | 10.01 | 9.58 | 8.94 |
| 128, 100 000 | 57.22 | 53.41 | 49.59 |
| 768, 1 024 | 53.41 | 53.41 | 53.41 |
| 768, 100 000 | 283.12 | 283.12 | 283.12 |

(Coefficient of variation 1.7–6.8%, except Inlined 768/1 024 at 12%. Identical medians are real:
iteration counts were the same, and the timer resolution bounds the reported value.)

## 3. HNSW and Flat, scalar vs AVX2 on the same index file (prefetch off)

Recall and distance computations are identical for both tiers (same graph, ties within rounding did
not change any result in these runs), so QPS is compared at matched recall.

| d | ef | recall@10 | dist. comps | scalar QPS | AVX2 QPS | speedup | scalar p99 | AVX2 p99 |
|---|---|---|---|---|---|---|---|---|
| 128 | 10 | 0.708 | 388 | 14 138 | 33 268 | 2.35× | 114 µs | 46 µs |
| 128 | 32 | 0.938 | 660 | 7 513 | 15 896 | 2.12× | 177 µs | 95 µs |
| 128 | 64 | 0.991 | 874 | 5 384 | 10 119 | 1.88× | 246 µs | 142 µs |
| 128 | 128 | 1.000 | 1 039 | 4 149 | 6 934 | 1.67× | 303 µs | 186 µs |
| 128 | 512 | 1.000 | 1 132 | 2 634 | 3 579 | 1.36× | 447 µs | 342 µs |
| 768 | 16 | 0.740 | 484 | 2 726 | 7 420 | 2.72× | 458 µs | 201 µs |
| 768 | 64 | 0.966 | 881 | 1 442 | 3 520 | 2.44× | 846 µs | 395 µs |
| 768 | 128 | 0.991 | 1 043 | 1 181 | 2 770 | 2.35× | 969 µs | 494 µs |
| 768 | 256 | 0.994 | 1 115 | 1 052 | 2 336 | 2.22× | 1 264 µs | 568 µs |

| Exact (Flat) search, 100 000 rows | scalar QPS | AVX2 QPS | speedup |
|---|---|---|---|
| d = 128 | 131.2 | 390.0 | 2.97× |
| d = 768 | 18.4 | 67.3 | 3.66× |

| HNSW build, 100 000 × 128 | scalar | AVX2 |
|---|---|---|
| `Collection::add_batch` | 40.9 s | 17.9 s (2.28×) |

(The two builds produce different graphs: the tiers round differently.) AVX2 build at d = 768: 42.3 s.

## 4. Prefetch ablation (AVX2 unless noted)

| d | ef | QPS off | QPS on | change |
|---|---|---|---|---|
| 128 | 10 / 64 / 128 / 512 | 33 268 / 10 119 / 6 934 / 3 579 | 33 233 / 10 377 / 6 933 / 3 643 | −0.1% / +2.5% / 0.0% / +1.8% |
| 768 | 16 / 64 / 128 / 256 | 7 420 / 3 520 / 2 770 / 2 336 | 8 372 / 4 052 / 3 080 / 2 491 | +12.8% / +15.1% / +11.2% / +6.6% |
| 128 scalar | 10 / 64 / 512 | 14 138 / 5 384 / 2 634 | 15 340 / 5 806 / 2 617 | +8.5% / +7.8% / −0.6% |
| 768 scalar | 16 / 64 / 256 | 2 726 / 1 442 / 1 052 | 2 918 / 1 539 / 1 116 | +7.0% / +6.7% / +6.1% |

Build ablation (two runs each, ef = 64 queries on the built graph):

| d | build off | build on | QPS off | QPS on |
|---|---|---|---|---|
| 128 | 17.33 / 17.12 s | 17.62 / 17.37 s (+1.5%) | 10 295 / 10 425 | 10 302 / 10 585 (+0.8%) |
| 768 | 42.68 / 42.49 s | 40.65 / 40.54 s (−4.7%) | 3 498 / 3 436 | 4 053 / 4 008 (+16.3%) |

## Observations (this machine and compiler)

1. **Hand-written AVX2 beats the compiler.** At d ≥ 100, `avx2_acc4` is 1.35–2.4× faster than MSVC's
   `/fp:fast` auto-vectorised loop and 6–18× faster than the strict scalar oracle. Below d = 32 it is
   slower than both (8 dims: 4.06 vs 2.86 ns autovec): the setup of four accumulators and the
   horizontal reduction dominate. Configured dimensions in practice are ≥ 100.
2. **Four accumulators matter** (the DESIGN §10.3 hypothesis holds): `acc4` is 1.9× faster than `acc1`
   at d = 128 and 2.9× at d = 768 cache-hot. The loop-carried dependency through one accumulator
   limits `acc1` to about one fused-multiply-add chain per 8 floats.
3. **The masked tail does not pay off**: identical at dims divisible by 32, 5% faster at d = 100,
   1.2–1.25× slower at d = 8/16. `acc4` with a scalar tail is the `avx2` tier.
4. **Input alignment matters for AVX2 here**: inputs starting one float after a 64-byte boundary are
   15% slower at d = 128 and 31% slower at d ≥ 384 (not for scalar). The cause was not investigated. Stored rows are 64-byte
   aligned in heap chunks and at multiples of `dim × 4` bytes from a page boundary in mapped files;
   query buffers are the caller's.
5. **Batch kernels hit memory bandwidth.** Cache-hot, 1 024 × 768 runs 12.6× faster than scalar; over
   100 000 rows (293 MiB) only 3.8×, and `acc1` ≈ `acc4`. Flat search follows the cold number: 3.66×
   at d = 768 and 2.97× at d = 128. The Phase 1 question is answered: MSVC does not vectorise
   `scalar_autovec::l2sq_1_to_n` (its disassembly is almost entirely scalar `ss` instructions; the
   pairwise function uses packed `ps` instructions), so the autovec batch equals scalar.
6. **Amdahl gap for HNSW.** Estimated from cache-hot kernel cost × distance computations: at d = 128,
   ef = 64 the scalar kernels account for ~63 µs of the 186 µs mean query (34%), predicting 1.44× from
   AVX2; the measured speedup is 1.88×, so the in-search scalar cost is higher than the cache-hot
   estimate (not explained without a profiler). At d = 768 the estimate is 466 of 693 µs (67%),
   predicting 2.72×; measured 2.44×, below the prediction because rows are cache-cold. Kernel speedups
   of 9–16× become 1.4–2.7× end-to-end queries.
7. **Dispatch cost is small but visible at d = 128.** The production path (switch + indirect call) costs
   about 1.1 ns more per distance than an inlined AVX2 loop cache-hot (10.0 vs 8.9 ns) and 7.6 ns
   memory-bound; at d = 768 there is no difference. At 874 distances per query that is at most
   ~1–7 µs of a ~99 µs AVX2 query (1–7%), below the prefetch and kernel effects and within run-to-run
   variation of the macro runs; table dispatch stays (ADR-0002). This loop has no visited-set or heap
   work, so the real share is lower than the experiment's.
8. **Prefetch helps at large dimensions:** +7–16% QPS and −4.7% build time at d = 768; at d = 128 the
   effect (−0.1…+2.5% QPS with AVX2, +1.5% build time) is within single-run variation. It is on by
   default. Only the first cache line of each row is prefetched; prefetching more lines is untested.
9. Single runs on a laptop; GCC/Clang, other CPUs, cosine/IP metrics and multi-threaded runs were not
   measured.
