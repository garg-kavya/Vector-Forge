# ADR-0002: Runtime SIMD dispatch through a process-wide kernel table

- **Status:** Accepted
- **Date:** 2026-09-14
- **Related:** docs/DESIGN.md §10.4–10.6, docs/simd.md

## Context

Distance kernels dominate exact search and are a large share of HNSW query time. The development
machine (Zen 2) and every GitHub-hosted runner support AVX2 + FMA, but release binaries, Docker
images and Python wheels must also run on x86-64 CPUs without them, and the library must build for
non-x86 targets. Binaries compiled with `-march=native` or `/arch:AVX2` crash with an illegal
instruction on such CPUs. Correctness constraints: HNSW graphs are compared byte for byte in
determinism tests, FlatBackend and HNSW must compute identical distances, and a user who asks for a
tier must never be silently given another.

## Decision

1. Generic code is compiled for the baseline instruction set. The AVX2 + FMA kernels live in one
   translation unit, `src/simd/kernels_avx2.cpp`, built as the object library `vf_simd_avx2` with
   per-target flags (`/arch:AVX2`; `-mavx2 -mfma -ffp-contract=off`, plus unaligned vector moves on
   MinGW). Its helpers have internal linkage (`simd/avx2_kernels_inline.hpp`); it exports only
   `vf::detail::avx2` functions.
2. A `KernelTable` of function pointers is selected once per process (thread-safe static
   initialisation) from CPUID/XGETBV and the `VF_SIMD` environment variable (`auto`, `scalar`,
   `avx2`). Requests only downgrade; an invalid or unsupported request is an error returned by
   `vf::simd_status()`, `Collection::create/load`, `vf::distance`, `vf::normalize` and every CLI
   command, and the table stays scalar.
3. Backends store a pointer to the table; each distance is a switch on the score mode plus one
   indirect call. The search loop is not instantiated per instruction set.
4. The `avx2` tier uses the four-accumulator kernel with a scalar tail (`acc4`), chosen from the
   variant benchmarks. `acc1` and `acc4_masked` stay compiled for benchmarks and tests.
5. `tools/check_isa_leak.py` runs in CI on GCC, Clang and MSVC release builds and fails if AVX
   instructions appear outside the AVX2 object (toolchain-guarded dispatch excepted) or if the AVX2
   object exports any other function.

## Alternatives considered

| Alternative | Why not chosen |
|---|---|
| Compile everything for AVX2 (`VF_NATIVE`) | Illegal-instruction crashes on older CPUs; kept only as a benchmark configuration |
| GCC/Clang `target_clones` / function multiversioning | Not available on MSVC; ifunc resolvers are ELF-only |
| Per-ISA instantiation of the whole search loop (template on the kernel) | The dispatch experiment measured 1.1 ns (cache-hot) to 7.6 ns (memory-bound) per distance at d = 128, at most ~7% of a query, and nothing at d = 768 (docs/simd.md "Choosing the defaults"); it would multiply the code compiled with AVX2 flags and the ISA-leak surface |
| Per-collection kernel choice | No use case; one process-wide tier keeps results labelled consistently (`stats().simd`) |
| Silently falling back when `VF_SIMD=avx2` is unsupported | Benchmark and test results would be mislabelled |

## Consequences

- One binary runs everywhere; `VF_SIMD=scalar` gives A/B comparisons with the same binary and index
  file, and CI runs the full test suite on both tiers.
- Results differ between tiers within floating-point rounding (reduction order), so determinism is
  claimed per (tier, single thread): golden graph fingerprints are recorded per tier. Within a tier
  every kernel is bit-identical across alignments and across pairwise and 1-to-N calls, and
  symmetric in its arguments.
- Adding a tier (AVX-512, NEON) means one more object library, table and fingerprint set.
- The tier is fixed for the process lifetime; changing `VF_SIMD` after the first kernel use has no
  effect.

## Validation

- Tests: `tests/unit/test_kernels.cpp` (all tables, dims 0–67 and 100–1543, 8 offsets, symmetry,
  batch = pairwise), `tests/unit/test_dispatch.cpp`, CTest `simd.*` runs with `VF_SIMD` set,
  `HnswDeterminism.GoldenFingerprint` per tier, CI "Test (scalar tier)" and "ISA leak check" steps.
- Benchmarks: `benchmarks/results/2026-09-14_ryzen7-4800h_msvc-release_phase5/` (kernel variants,
  dispatch experiment, macro scalar vs AVX2, prefetch ablation).
