# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses semantic versioning.

## [Unreleased]

### Added
- Phase 0: repository scaffolding, CMake build with presets (MSVC, MinGW, Linux GCC/Clang,
  sanitizers), pinned GoogleTest and Google Benchmark, warning and sanitizer modules,
  `tools/dev-env.ps1`, clang-format/clang-tidy configuration, CI and lint workflows,
  engineering design document.
- Phase 1: `Status`/`Result<T>` error handling; core types (`Metric`, `IndexType`, ids, `Neighbor`)
  and validated `HnswParams`/`CollectionConfig`; overflow-checked arithmetic; deterministic
  SplitMix64/xoshiro256** RNG with fully specified float conversions; 64-byte aligned allocation;
  scalar reference distance kernels (dot, squared L2, norm, 1-to-N) plus compiler-autovectorised
  comparison variants; x86 CPU feature detection (CPUID + XGETBV); kernel dispatch table (scalar);
  input validation, normalisation and public `vf::distance`/`vf::normalize`; chunked append-only
  `VectorStore` with stable row addresses; unit tests; Google Benchmark scalar kernel baseline.
