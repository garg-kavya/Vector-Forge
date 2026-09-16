#!/usr/bin/env python3
"""Python binding overhead (docs/DESIGN.md §21 Phase 8).

Searches one saved index with the same query file from C++ (`vf_bench --scenario threads`, one
thread) and from Python (one batch call, and one call per query), and writes a JSON report.

    python benchmarks/python/bench_overhead.py --vf-bench out/build/msvc-release/benchmarks/vf_bench.exe \\
        --index hnsw_d128.vfidx --queries queries.npy --out python_overhead.json
"""

import argparse
import json
import platform
import subprocess
import time

import numpy as np

import vectorforge as vf


def median_qps(fn, count, repeat):
    rounds = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        rounds.append(count / (time.perf_counter() - start))
    rounds.sort()
    return rounds[len(rounds) // 2], rounds[0], rounds[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vf-bench", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--queries", required=True, help=".npy float32 (n, d)")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    queries = np.load(args.queries).astype(np.float32, copy=False)
    nq = len(queries)

    cpp = subprocess.run(
        [args.vf_bench, "--scenario", "threads", "--index-file", args.index,
         "--queries-file", args.queries, "--threads", "1", "--repeat", str(args.repeat),
         "--k", str(args.k), "--ef-search", str(args.ef_search)],
        check=True, capture_output=True, text=True).stdout
    cpp_report = json.loads(cpp)
    cpp_qps = cpp_report["results"][0]["qps_median"]

    index = vf.Index.load(args.index, mmap=False)
    index.search(queries, k=args.k, ef_search=args.ef_search, num_threads=1)  # warm-up
    batch = median_qps(lambda: index.search(queries, k=args.k, ef_search=args.ef_search,
                                            num_threads=1), nq, args.repeat)

    def per_query():
        for q in queries:
            index.search(q, k=args.k, ef_search=args.ef_search, num_threads=1)

    single = median_qps(per_query, nq, args.repeat)

    report = {
        "schema": 1,
        "suite": "python-overhead",
        "vectorforge": {"version": vf.__version__, "git_sha": vf.git_sha, "simd": vf.simd_level()},
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpp": {"git": cpp_report["git"], "build": cpp_report["build"],
                "machine": cpp_report["machine"]},
        "params": {"index": cpp_report["params"], "queries": nq, "k": args.k,
                   "ef_search": args.ef_search, "repeat": args.repeat, "threads": 1},
        "results": {
            "cpp_search_batch_qps": cpp_qps,
            "python_batch_qps": {"median": batch[0], "min": batch[1], "max": batch[2]},
            "python_per_query_qps": {"median": single[0], "min": single[1], "max": single[2]},
            "python_batch_vs_cpp": batch[0] / cpp_qps,
            "python_per_query_overhead_us": (1.0 / single[0] - 1.0 / batch[0]) * 1e6,
        },
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps(report["results"], indent=2))


if __name__ == "__main__":
    main()
