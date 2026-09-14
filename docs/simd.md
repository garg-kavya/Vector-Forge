# SIMD kernels and runtime dispatch

VectorForge computes distances with one of several kernel tiers, selected once per process at
runtime. This page describes the tiers as built in Phase 5, how selection works, what guarantees each
tier gives, and what the measurements on the development machine showed. Design rationale:
[DESIGN.md §10](DESIGN.md#10-simd-design) and [ADR-0002](adr/0002-runtime-simd-dispatch.md).

Source: `src/simd/` (`kernels.hpp`, `kernels_scalar.cpp`, `kernels_scalar_autovec.cpp`,
`kernels_avx2.cpp`, `avx2_kernels_inline.hpp`, `dispatch.{hpp,cpp}`, `cpu_features.cpp`),
`cmake/SimdFlags.cmake`, `tools/check_isa_leak.py`.

## Tiers

| Tier | Kernels | Selected when |
|---|---|---|
| `avx2` | AVX2 + FMA, four accumulators, scalar tail (`avx2_acc4`) | the build contains the AVX2 object and the CPU reports AVX, AVX2, FMA and the OS saves YMM state (XGETBV) |
| `scalar` | strict left-to-right reference loops, no FP contraction | otherwise, or `VF_SIMD=scalar` |

Benchmark-only tables (never selected): `scalar_autovec` (the same loops with compiler
auto-vectorisation allowed), `avx2_acc1` (one accumulator) and `avx2_acc4_masked` (masked-load tail).

```text
VF_SIMD unset / "" / auto   best tier available
VF_SIMD=scalar              scalar reference kernels
VF_SIMD=avx2                AVX2 kernels, or an error if they cannot run
anything else               error
```

The request can only downgrade. An invalid or unsupported request never silently selects another
tier: `vf::simd_status()` returns `INVALID_ARGUMENT` or `FAILED_PRECONDITION`, and so do
`Collection::create`, `Collection::load`, `vf::distance`, `vf::normalize` and every `vectorforge`
command. The tier in use is reported by `vf::active_simd_level()`, `CollectionStats::simd`,
`vectorforge info`, and the JSON context of every benchmark (`vf_simd_level`, `simd_tier`,
`kernel_table`). The environment variable is read on first kernel use; changing it later has no
effect in that process.

Build options: `VF_ENABLE_AVX2=OFF` (or a non-x86-64 target) omits the AVX2 object; `VF_NATIVE=ON`
compiles everything for the host CPU and is for benchmarking only.

## AVX2 kernel structure

For `l2sq` (`dot` and `norm2` are analogous, `norm2(a) = dot(a, a)`):

```text
acc0..acc3 = 0
for each 32-float block:          acc_j = fmadd(a_j - b_j, a_j - b_j, acc_j)   j = 0..3 (8 floats each)
for each remaining 8-float block: acc0  = fmadd(diff, diff, acc0)
[acc4_masked: the < 8 float tail is read with _mm256_maskload_ps into acc1]
s = horizontal_sum((acc0 + acc1) + (acc2 + acc3))     # lanes ((l0+l4)+(l2+l6)) + ((l1+l5)+(l3+l7))
s += scalar tail, left to right
```

- Inputs are loaded with `loadu`: stored rows are 64-byte aligned, but query buffers need not be.
- The 1-to-N kernels call the same inlined pairwise implementation for each row.
- The AVX2 translation unit is compiled with `-ffp-contract=off` (GCC/Clang) so the scalar tail is
  not fused; MSVC's `/fp:precise` does not contract. MinGW adds `-Wa,-muse-unaligned-vector-move`
  because GCC on Windows does not keep the stack 32-byte aligned (GCC bug 54412).

## Numerical guarantees

- **Within a tier, results are exact functions of the inputs.** Every kernel has a fixed reduction
  order, so it returns bit-identical values for any input alignment, the 1-to-N kernels return exactly
  the pairwise values (FlatBackend and HNSW distances stay identical), and `dot(a, b) == dot(b, a)`
  and `l2sq(a, b) == l2sq(b, a)` bit for bit (HNSW construction relies on symmetry).
- **Across tiers, results differ within rounding.** Reduction orders differ, so the tiers agree only
  to within the error bound of summing `n` float terms in any order; `test_kernels.cpp` checks every
  table against a double-precision reference with the rigorous bound
  `((1 + ε)^(n+4) − 1) · Σ|terms|` for d = 0…67, 100, 128, 384, 768, 1536, 1537, 1543 at eight input
  offsets.
- **Determinism is claimed per (tier, single thread).** A graph built on one tier can differ from one
  built on the other: normalised vectors and comparisons near ties round differently. The golden HNSW
  fingerprints are recorded per tier; on the test dataset the L2 graph is identical on both tiers and
  the cosine graph is not. Recall and reachability measurements in `test_hnsw_recall.cpp` were
  identical on both tiers. Index files do not record the tier; any file loads on any tier.

## ISA isolation

Generic code is compiled for baseline x86-64. Only `kernels_avx2.cpp` (object library
`vf_simd_avx2`) gets AVX2 flags. If an inline function or template were instantiated both there and
in a generic source, the linker could keep the AVX2 copy for every caller and crash older CPUs, so:

- `kernels_avx2.cpp` includes only `kernels.hpp` (declarations; `vectorforge/simd_level.hpp` is
  dependency-free), `avx2_kernels_inline.hpp` (internal-linkage templates, `#error` without
  `__AVX2__`) and standard/intrinsic headers, and exports only `vf::detail::avx2` functions.
- `tools/check_isa_leak.py <build-dir>` disassembles the library and CLI objects (objdump or dumpbin)
  and fails on any VEX/EVEX instruction outside the AVX2 object, on an AVX2 object without YMM code,
  or on an AVX2 object exporting any other function. MSVC's auto-vectorizer emits AVX2 loops guarded
  by `__isa_available`, and the UCRT's inline `wmemcmp` tests `_Avx2WmemEnabled`; functions with
  those runtime guards are allowed and counted. A `VF_NATIVE=ON` build fails the check as expected.
- CI runs the check on GCC 14, Clang 18 and MSVC release builds, requires the AVX2 tier on the
  runner (`VF_SIMD=avx2 vectorforge gen-data ...`), and runs the full test suite a second time with
  `VF_SIMD=scalar`. CTest also runs the tier-sensitive tests with `VF_SIMD` set to `scalar`, `avx2`
  and an invalid value (`simd.*`, `cli.simd_invalid_request`).

## Measurements

Result files: [`benchmarks/results/2026-09-14_ryzen7-4800h_msvc-release_phase5/`](../benchmarks/results/2026-09-14_ryzen7-4800h_msvc-release_phase5/README.md)
(AMD Ryzen 7 4800H, Zen 2; MSVC 19.50 `msvc-release`; single thread; laptop on AC power). These are
one machine and one compiler; GCC and Clang were not benchmarked.

### Kernels

Median ns per call, cache-hot, aligned inputs (`l2sq`):

| dim | scalar | scalar_autovec (MSVC `/fp:fast`) | avx2_acc4 (tier) | avx2_acc1 | avx2_acc4_masked |
|---|---|---|---|---|---|
| 8 | 3.34 | 2.86 | 4.06 | 3.59 | 5.00 |
| 128 | 72.55 | 13.44 | 7.79 | 14.63 | 8.64 |
| 768 | 528.84 | 75.92 | 32.17 | 94.05 | 32.22 |
| 1536 | 1067.80 | 145.86 | 60.49 | 206.53 | 60.49 |

- Hand-written AVX2 beats the auto-vectoriser by 1.35–2.4× and the scalar oracle by 6–18× at d ≥ 100.
  Below d = 32 it is slower than both (accumulator setup and reduction dominate).
- Four accumulators beat one by 1.9× (d = 128) and 2.9× (d = 768), confirming the loop-carried
  dependency hypothesis of DESIGN §10.3.
- The masked-load tail gains 5% at d = 100 and loses 23% at d = 8/16; it is not used.
- Inputs one float past a 64-byte boundary ran 15% (d = 128) to 31% (d ≥ 384) slower on AVX2; not
  investigated.
- MSVC does not vectorise `scalar_autovec::l2sq_1_to_n` (scalar `ss` instructions after inlining the
  pairwise loop into the row loop), which answers the open Phase 1 observation.

### Choosing the defaults

| Decision | Data | Choice |
|---|---|---|
| AVX2 variant | acc4 fastest or within 5% (masked tail at d = 100) at every d ≥ 100; masked tail slower below 32 | `avx2_acc4` |
| Dispatch level | table call costs +1.1 ns/distance cache-hot, +7.6 ns memory-bound at d = 128 vs an inlined AVX2 loop; 0 at d = 768; ≤ 1–7% of a d = 128 query | keep the table; no per-ISA search loop |
| HNSW prefetch | d = 768: +7–16% QPS, −4.7% build; d = 128: within ±2.5% | `HnswSearchOptions::prefetch = true` |

### End to end

Same index file, same binary, `VF_SIMD=scalar` vs `avx2` (recall identical):

| Workload | scalar | AVX2 | speedup |
|---|---|---|---|
| HNSW 100K × 128, ef = 64 (recall 0.991) | 5 384 QPS | 10 119 QPS | 1.88× |
| HNSW 100K × 768, ef = 64 (recall 0.966) | 1 442 QPS | 3 520 QPS | 2.44× |
| Flat 100K × 128 | 131 QPS | 390 QPS | 2.97× |
| Flat 100K × 768 | 18.4 QPS | 67.3 QPS | 3.66× |
| HNSW build 100K × 128 | 40.9 s | 17.9 s | 2.28× |

**The Amdahl gap.** Kernel speedups of 9× (d = 128) and 16× (d = 768) become 1.4–2.7× for HNSW
queries and 3–3.7× for Flat scans. Flat scans are memory-bandwidth-bound: over 100 000 rows the
batch kernel is only 3.8× faster than scalar at d = 768 (12.6× cache-hot). For HNSW, cache-hot kernel
cost × distance computations estimates the scalar kernels at 34% of the query time at d = 128 and
67% at d = 768 (predicting 1.44× and 2.72×); measured 1.88× and 2.44×. No profiler was run, so the
remaining discrepancy (in-search scalar cost above the cache-hot estimate at d = 128, cache-cold rows at
d = 768) is not attributed further.

### Prefetch

`HnswSearchOptions::prefetch` issues a `_mm_prefetch`/`__builtin_prefetch` hint for the first cache
line of every unvisited neighbour's row before computing distances to a node's neighbour list (query
and construction searches). It never changes results (`HnswDeterminism.PrefetchDoesNotChangeGraphOrResults`).
Prefetching more than the first line of large rows is untested.