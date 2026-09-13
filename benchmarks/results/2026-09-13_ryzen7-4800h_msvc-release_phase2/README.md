# Phase 2 exact-search baselines — 2026-09-13

Raw data (Google Benchmark JSON, aggregates only; host name and local executable path redacted):

- [`topk.json`](topk.json) — top-k selection strategies, 3 repetitions per case
- [`flat_search.json`](flat_search.json) — single-query Flat search latency, 5 repetitions per case

## Environment

| Item | Value |
|---|---|
| CPU | AMD Ryzen 7 4800H (Zen 2), 8 cores / 16 threads, AVX2+FMA, no AVX-512 |
| OS | Windows 11 Home 10.0.26200 |
| Power | AC power, battery 100%, custom power plan (laptop; thermal throttling not controlled) |
| Compiler | MSVC 19.50.35728, `msvc-release` preset (`/O2`, no `/arch`, no LTO) |
| Library | Google Benchmark 1.9.5 |
| SIMD tier | scalar |
| Source | JSON `vf_git_sha` = `7790ee47ecc6-dirty`: built from the working tree that became the Phase 2 commit (parent `7790ee4` plus the Phase 2 changes); no source changes were made after the run other than adding these result files and documentation |

## Reproduce

```powershell
. .\tools\dev-env.ps1
cmake --preset msvc-release
cmake --build --preset msvc-release
$b = "out\build\msvc-release\benchmarks\vf_micro_bench.exe"
& $b --benchmark_filter=BenchTopK --benchmark_repetitions=3 --benchmark_min_time=0.2s `
  --benchmark_report_aggregates_only=true --benchmark_out=topk.json --benchmark_out_format=json
& $b --benchmark_filter=BenchFlatSearch --benchmark_repetitions=5 --benchmark_min_time=0.5s `
  --benchmark_report_aggregates_only=true --benchmark_out=flat_search.json --benchmark_out_format=json
```

## Top-k selection (median time per selection)

Stream of n (distance, id) candidates with ascending ids, as produced by the Flat scan. `random` =
uniform distances (few candidates beat the current worst); `descending` = every candidate beats it
(adversarial). Insertion and heap work in k-sized storage; nth_element and partial_sort sort an n-sized
scratch copy (not allocation-free, reference only).

| n | pattern | k | insertion | heap | nth_element | partial_sort |
|---|---|---|---|---|---|---|
| 10 000 | random | 1 | 8.5 µs | **5.2 µs** | 91.9 µs | 13.4 µs |
| 10 000 | random | 10 | 9.4 µs | **6.6 µs** | 91.4 µs | 14.5 µs |
| 10 000 | random | 100 | 28.5 µs | **15.8 µs** | 94.9 µs | 21.6 µs |
| 10 000 | random | 1000 | 748 µs | 209 µs | **131 µs** | 213 µs |
| 1 000 000 | random | 10 | 857 µs | **516 µs** | 9 644 µs | 1 870 µs |
| 1 000 000 | random | 100 | 918 µs | **560 µs** | 9 824 µs | 1 755 µs |
| 1 000 000 | random | 1000 | 2 789 µs | **1 062 µs** | 9 744 µs | 2 205 µs |
| 1 000 000 | descending | 10 | 5 747 µs | 9 790 µs | **2 956 µs** | 5 972 µs |
| 1 000 000 | descending | 1000 | 492 700 µs | 45 820 µs | **2 935 µs** | 43 640 µs |

(The JSON has all combinations of n ∈ {10⁴, 10⁶}, k ∈ {1, 10, 32, 64, 100, 1000} and both patterns.)

## Flat search latency (single query, k = 10, uniform data, scalar kernels)

| n | dim | L2 median | L2 QPS | Cosine median | Cosine QPS |
|---|---|---|---|---|---|
| 10 000 | 128 | 0.775 ms | 1 306 | 0.745 ms | 1 365 |
| 10 000 | 768 | 5.45 ms | 184 | 5.34 ms | 188 |
| 100 000 | 128 | 7.66 ms | 131 | 7.33 ms | 137 |
| 100 000 | 768 | 54.4 ms | 18.3 | 53.4 ms | 18.5 |

## Observations and decisions (this machine and compiler only)

1. **Decision: FlatBackend uses the bounded max-heap for every k.** On scan-like (random) streams the
   heap was fastest in every allocation-free comparison, including k = 1, where the sorted-insertion
   array was expected to win. The sorted-insertion variant was removed from the library and remains
   only in `benchmarks/micro/bench_topk.cpp` as a reference.
2. The heap degrades on the adversarial descending stream (every candidate accepted); nth_element is
   best there but needs O(n) scratch memory, which would break the allocation-free query path. A scan
   over real data resembles the random stream far more than the descending one.
3. **Distance computation dominates exact search.** Heap selection over random candidates took 6.6 µs
   for 10⁴ and 516 µs for 10⁶ candidates (k = 10), so roughly tens of microseconds for 10⁵, while the 10⁵ × 768 scan takes about 54 ms; latency scales roughly linearly with
   n × dim (about 1.4 billion components/s, matching the scalar kernel baseline from Phase 1). SIMD
   kernels (Phase 5) and parallel scans (Phase 6) are therefore the relevant optimisations, not
   selection.
4. Cosine is slightly faster than L2 because normalised collections score with the dot kernel plus a
   scale, whereas L2 uses the squared-difference kernel.

These numbers are baselines for later comparisons on the same machine, not performance claims.
