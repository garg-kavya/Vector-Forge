# Full benchmark suite — 2026-09-17

Every scenario of DESIGN §16.3 that fits a 16 GB, 8-core laptop with synthetic data:
`benchmarks/configs/full.json`, 72 runs, no failed repetitions (1 h 25 min).

- [`manifest.json`](manifest.json): configuration, machine, source revision, start/end time and
  the list of scenarios that were **not** run, with reasons.
- `<run>/repN.json`: one file per repetition (fresh process each), host names and paths redacted.
- [`tables.md`](tables.md): every table (the README shows a subset);
  [`plots/`](plots): recall/QPS Pareto fronts, exact throughput, thread scaling, build scale,
  latency during ingestion.

## Environment and protocol

AMD Ryzen 7 4800H (8 cores / 16 threads, AVX2), 15.4 GB RAM, Windows 11 (10.0.26200), AC power,
MSVC 19.50 `msvc-release` build of `ceff91efdfdc-dirty` (the working tree that became the Phase 9
commit; library code identical to `ceff91e`), CPython 3.14.3 for the driver. The machine was an
ordinary desktop session: other desktop applications and two WSL polling loops were
running. No thread pinning. All file reads are warm-cache.

Data: seeded Gaussian mixtures (100 clusters, spread 0.1) from `vf_bench`; ground truth by exact
search. Repetition counts per run are in the configuration (1 for long builds, 3 otherwise; the
exact and threads scenarios repeat passes inside each process and report the median).

## Findings

**Exact search.** AVX2 is 2.4–3.8× faster than the scalar tier on one thread for 10 K–1 M rows.
Multi-threaded Flat search stops scaling once the scan no longer fits the cache: at 10 K × 128-d,
16 threads give 7.1× (AVX2); at 100 K and 1 M rows, 1.8–2.9× — memory bandwidth, not the kernels,
is the limit, and AVX2 at 16 threads is barely ahead of scalar at 16 threads for 100 K rows.

**Thread scaling above 12 threads is not reliable on this machine.** The Flat run shows 6.1× at
12 threads and 2.7× at 16. A follow-up probe on the same binary and a fresh 100 K × 128-d file gave
2 163 QPS at 12 threads and 880–1 580 QPS at 13–16 threads, but 2 063 QPS for a second 13-thread
pass in the same process: the drop depends on what else the desktop is doing, not on the thread
count. `parallel_for` splits a batch into fixed chunks, so one descheduled worker delays the whole
batch once every logical CPU is busy. Numbers up to 8–12 threads are the meaningful ones;
pinning (not implemented) and an idle machine are needed for the rest.

**HNSW parameters (100 K × 128-d, k = 10).** Larger M reaches a recall target with a smaller
ef: M = 32 gives 11 400 QPS at recall ≥ 0.99 against 10 350 for M = 16 and 5 500 for M = 8, at
1.9× the graph memory and 1.8× the build time of M = 16. ef_construction above 200 buys almost
nothing here. M = 48 is slower than M = 32 at recall ≥ 0.99 (more distance computations per hop).
For k = 100 the effective beam is max(ef, k), so every row reads "ef 10".

**Scale (M = 16, efC = 200, 16 build threads).** Recall at a fixed ef = 64 falls with N:
0.99 at 100 K × 128-d, 0.79 at 1 M × 128-d, 0.67 at 1 M × 768-d. ef must grow with the
collection: 1 M × 128-d reaches 0.915 at ef = 128 and 0.990 at ef = 512. Unreachable level-0 nodes
rise to 108 of 1 M (128-d) and 1 063 of 1 M (768-d) with concurrent builds; searches cannot return
those rows, which caps recall. Peak memory is dominated by the vectors (5.98 GiB at 1 M × 768-d;
the peak includes the ground-truth copy, which is freed before the searches).

**Build threads.** 7.4× at 16 threads for the 100 K × 128-d build (3.7× at 4, 5.7× at 8); the
concurrent (Level B) path at one thread matches the coarse path within 4%.

**Ingestion.** With one writer, Level B keeps the median search latency during inserts at
0.087 ms (idle 0.072 ms) against 2.28 ms under the coarse lock; four Level B writers insert
15 500 vectors/s with a p99.9 search latency of 0.42 ms.

**Storage.** Opening the 1 M × 768-d Flat file (2.9 GiB) takes 3.7 s into the heap and 0.17 s
mapped (+81 MiB RSS instead of +3 GiB); steady-state QPS is the same once pages are in memory.
Prefaulting doubles the open time without improving the first query on a warm cache.

**SIMD in HNSW (100 K × 768-d).** AVX2 is 2.3–2.7× the scalar tier across ef 16–256 with
identical recall. **Prefetch** is worth 9–12% QPS at ef ≥ 32 (one noisy repetition at ef 16).
**Neighbour selection:** the simple (closest-M) rule builds 1.3× faster but caps recall at 0.86
(1 548 unrepaired orphans, 112 unreachable nodes); the heuristic reaches 0.99 at ef = 64.

**HTTP (100 K × 128-d HNSW, loopback, one client).** Median in-process query 0.11 ms, server
handling 0.14 ms, full round trip 0.32 ms; JSON ingestion 18 600 vectors/s against 419 000 for the
binary endpoint.

## Not run

See `not_run` in the manifest and the table in [`tables.md`](tables.md): 1 M × 1536-d (RAM), real
datasets (downloads need approval; `tools/datasets/fetch.py` prepares them), cold-cache loads,
thread pinning, the 12-point sweep at 1 M rows, and hnswlib/FAISS reference runs.

## Reproduce

```powershell
cmake --preset msvc-release
cmake --build --preset msvc-release --target vf_bench vf_http_bench vf_micro_bench
python benchmarks\scripts\run_suite.py benchmarks\configs\full.json --build-dir out\build\msvc-release --out results\my-run
python benchmarks\scripts\make_readme_tables.py results\my-run
python benchmarks\scripts\plot_results.py results\my-run
```

Methodology: [docs/benchmarking.md](../../../docs/benchmarking.md).
