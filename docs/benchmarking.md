# Benchmarking

How VectorForge is measured, how to reproduce the published numbers, and what was not measured.
Plan: [DESIGN.md §16](DESIGN.md#16-benchmark-strategy-o). Results live in
[`benchmarks/results/`](../benchmarks/results/); the complete suite run is
[`2026-09-17_ryzen7-4800h_msvc-release_suite`](../benchmarks/results/2026-09-17_ryzen7-4800h_msvc-release_suite/README.md).

## Rules

1. No number is published without the raw result file it comes from, in the same directory as the
   text that quotes it, together with the machine, build and source revision.
2. Every macro configuration runs in its own process (`run_suite.py` starts one process per
   repetition), after a warm-up pass inside that process.
3. Distributions, not single values: latency percentiles from all samples, throughput as the median
   of several passes (min–max kept); repeated processes where the time budget allows (see
   "Repetitions").
4. Every approximate result is compared with exact search on the same queries.
5. Anything that could not be run on the machine is listed as "not run" with the reason.
6. Result files contain no host names, user names or local paths (`run_suite.py` redacts Google
   Benchmark's `host_name` and `executable`; `tests/test_harness.py` checks it).

## Tools

| Tool | Measures |
|---|---|
| `vf_bench` (benchmarks/macro/vf_bench.cpp) | one scenario per process: `sweep` (HNSW build + ef_search sweep, recall against exact results, per-query latency, distance computations), `exact` (Flat throughput and latency), `build` (parallel HNSW construction, validator, recall), `threads` (batch search scaling), `ingest` (search latency during `add_batch`), `save` / `load` (persistence) |
| `vf_micro_bench` (Google Benchmark) | distance kernels, top-k strategies, visited sets, dispatch overhead |
| `vf_http_bench` | HTTP round trip vs in-process search, JSON vs binary ingestion |
| `benchmarks/python/bench_overhead.py` | Python binding overhead |
| `benchmarks/scripts/run_suite.py` | runs a JSON configuration, one process per repetition, writes a run directory with a manifest |
| `benchmarks/scripts/make_readme_tables.py` | Markdown tables from a run directory (and the README benchmark section) |
| `benchmarks/scripts/plot_results.py` | PNG plots from a run directory (matplotlib) |
| `tools/datasets/fetch.py`, `hdf5_to_npy.py` | public datasets (only with `--yes`) and conversion to `.npy` |

## Running

```powershell
. .\tools\dev-env.ps1
cmake --preset msvc-release; cmake --build --preset msvc-release
python benchmarks\scripts\run_suite.py benchmarks\configs\full.json --list
python benchmarks\scripts\run_suite.py benchmarks\configs\full.json `
    --build-dir out\build\msvc-release --out benchmarks\results\<date>_<machine>_<build>_suite --resume
python benchmarks\scripts\make_readme_tables.py benchmarks\results\<run> --readme README.md
python benchmarks\scripts\plot_results.py benchmarks\results\<run>
```

`--resume` continues an interrupted run (completed repetitions are kept); `--only REGEX` selects
runs. `benchmarks/configs/smoke.json` runs every scenario at toy size in seconds; the
`bench-smoke` workflow runs it together with the harness unit tests on every push (CI runner
numbers are not meaningful and are not published).

Configure the build right before a run: the source revision a tool reports (`git.sha`) is captured
at CMake configure time, while the manifest records the checkout's revision when the run started.

### Run directory

```text
manifest.json      suite, machine (OS, CPU counts, RAM, power plan), git revision, the configuration,
                   "not run" entries, and per repetition: status, return code, seconds
<run id>/rep1.json tool output (vf_bench schema 1 or Google Benchmark JSON, redacted)
tables.md, plots/  generated
README.md          environment notes and interpretation (written by hand)
```

A `vf_bench` result always has `schema`, `suite`, `git.sha`, `machine.cpu`, `build` (compiler,
build type, SIMD tier, kernel table), then scenario-specific `dataset`, `params` and result
sections. This is a per-scenario variant of the schema sketched in DESIGN §16.6: the machine block
of the manifest replaces the per-file machine details, and `null` placeholders are not written.

## Methodology

**Datasets.** Seeded synthetic data from `vf::detail::SyntheticGenerator`: a Gaussian mixture of
100 clusters with σ = 0.1 (seed 1 for base vectors, seed 2 for queries, so queries never coincide
with base vectors; the mixture centres are shared). Real datasets (SIFT1M, GloVe-100, embedding
sets) are supported by the tools but were not downloaded (see "Not run").

**Throughput.** `search_batch` over the whole query set with T threads (the caller plus T − 1
pool workers), timed with `steady_clock` after a warm-up pass; `--repeat` passes, median reported.

**Latency.** Each query issued individually on one thread and timed; all samples sorted;
percentiles by nearest rank (`ceil(p/100 · n)`-th smallest). p99.9 is only reported where the
sample count supports it (the ingest scenario reports it for completeness with ≥ 1 000 samples;
treat it as the maximum of the top 0.1 % there).

**Recall@k.** Tie-tolerant recall against exact results computed with the Flat index on the same
data (DESIGN §15.2); `recall_min` is the worst query. For each sweep, "QPS at recall ≥ X" is the
first ef_search (ascending) whose mean recall reaches X — no interpolation.

**Build time.** Wall time of the `add_batch` calls (or backend `add` calls in `sweep`), excluding
data generation. `sweep` builds serially through the backend so that graph statistics and distance
counters are exact; `build`/`scale` use the Concurrent mode with a pool.

**Memory.** Component accounting from `Collection::stats()` (`index_bytes` = graph + build scratch)
and peak resident set size (Windows `PeakWorkingSetSize`, Linux `VmHWM`) of the benchmark process,
which also holds the generated input and the exact-search copy — an upper bound for the index.

**Repetitions.** Fast scenarios run in 3 separate processes; the long ones (exact search on 10⁶
vectors, scale builds, the selection ablation) in one process with several passes inside where
applicable. This is fewer than the 5 processes DESIGN §16.5 asks for, to keep the complete suite
within about two hours on the laptop; the spread between repetitions is reported where it exists.

**Machine state.** Laptop on AC power, balanced/performance power plan as recorded in the
manifest, no other benchmark running, thermal state not controlled. No thread pinning. With 8
physical cores and 16 logical CPUs, results above 8 threads measure SMT.

## Not run

The complete list with reasons is in each run's `manifest.json` (`not_run`) and table. In short:
exact search and HNSW scale at 1M × 1536 (RAM), real datasets (downloads need approval), cold-cache
load timings (not possible on Windows without a reboot), thread pinning (not implemented), the 1M
parameter grid (serial builds would take hours; 1M is covered by the scale runs), external
libraries (optional, not installed).
