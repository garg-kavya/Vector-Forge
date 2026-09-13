# Phase 1 scalar kernel baseline — 2026-09-13

Raw data: [`kernels_scalar.json`](kernels_scalar.json) (Google Benchmark JSON, 5 repetitions per case,
aggregates only; host name and local executable path redacted).

## Environment

| Item | Value |
|---|---|
| CPU | AMD Ryzen 7 4800H (Zen 2), 8 cores / 16 threads, AVX2+FMA, no AVX-512 |
| OS | Windows 11 Home 10.0.26200 |
| Power | AC power, battery 100%, custom power plan (laptop; thermal throttling not controlled) |
| Compiler | MSVC 19.50.35728 (VS Build Tools 2026), `msvc-release` preset (`/O2`, no `/arch`, no LTO) |
| Library | Google Benchmark 1.9.5 |
| SIMD tier | scalar (`vf_simd_level` in JSON context) |
| Source | commit `9ba5aa828d45` (Phase 1), clean working tree; matches `vf_git_sha` in the JSON context. Run at 2026-09-13T18:44+05:30 |

## Reproduce

```powershell
. .\tools\dev-env.ps1
cmake --preset msvc-release
cmake --build --preset msvc-release
out\build\msvc-release\benchmarks\vf_micro_bench.exe --benchmark_filter="Bench(Dot|L2sq)" `
  --benchmark_repetitions=5 --benchmark_min_time=0.2s --benchmark_report_aggregates_only=true `
  --benchmark_out=kernels_scalar.json --benchmark_out_format=json
```

## Median wall time per call (aligned inputs)

| dim | dot scalar | dot autovec | l2sq scalar | l2sq autovec |
|---|---|---|---|---|
| 128 | 69.1 ns | 11.4 ns | 73.4 ns | 13.8 ns |
| 768 | 519.0 ns | 69.9 ns | 528.2 ns | 76.1 ns |
| 1536 | 1058.8 ns | 138.5 ns | 1068.2 ns | 144.3 ns |

`l2sq_1_to_n` over 1 024 rows: 73.7 µs (scalar) vs 73.9 µs (autovec) at d=128; 546.8 vs 549.8 µs at
d=768; 1136.2 vs 1121.2 µs at d=1536.

## Observations (this machine and compiler only)

1. **Scalar reference** processes roughly 1.4–1.8 billion components/s at d ≥ 128 — the expected cost of
   a strictly ordered float reduction the compiler may not vectorise.
2. **MSVC `/fp:fast` auto-vectorisation** of the pairwise kernels is several times faster than the strict
   scalar loop at d ≥ 100 (baseline SSE2 target). Any hand-written AVX2 kernel in Phase 5 must be compared
   against this variant, not only against the strict scalar oracle.
3. **The auto-vectorised batch kernel did not vectorise**: `scalar_autovec::l2sq_1_to_n` is no faster than
   scalar. Likely cause (not yet confirmed from the generated code): once the pairwise kernel is inlined
   into the row loop, MSVC no longer vectorises the inner reduction. This shows how fragile relying on
   auto-vectorisation is; Phase 5 will inspect the disassembly and re-measure.
4. Aligned vs one-float-offset inputs differ by at most 4.9% across all cases (median vs median), within
   the same range as differences between repeated runs of this suite on this laptop; no alignment effect
   is claimed.
5. GCC/Clang results may differ substantially (`-fopenmp-simd` path); they were not benchmarked in this run.

These numbers are a baseline for later comparisons on the same machine, not a performance claim.
