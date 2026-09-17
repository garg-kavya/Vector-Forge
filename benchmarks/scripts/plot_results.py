#!/usr/bin/env python3
"""Plots a run directory written by run_suite.py (needs matplotlib).

    python benchmarks/scripts/plot_results.py benchmarks/results/<run>

Writes PNG files into <run>/plots/: recall-QPS Pareto curves of the HNSW sweep, exact-search
throughput, build and search scaling with threads, build scaling with N and d, and ingest latency.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import load_results as lr  # noqa: E402

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    sys.exit("plot_results.py needs matplotlib (pip install matplotlib)")


def save(fig, out: pathlib.Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / name, dpi=120)
    plt.close(fig)
    print("wrote", out / name)


def plot_sweep(data, out):
    runs = lr.by_suite(data, "hnsw-sweep")
    grid = {k: v for k, v in runs.items() if "-k" not in k}
    if not grid:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    def params(run):
        return run["reps"][0]["params"]

    for run_id, run in sorted(grid.items(), key=lambda kv: (params(kv[1])["M"],
                                                            params(kv[1])["ef_construction"])):
        points = lr.merged_sweep(run["reps"])
        label = f"M={params(run)['M']} efC={params(run)['ef_construction']}"
        axes[0].plot([p["recall_at_k"] for p in points], [p["qps"] for p in points], marker="o",
                     markersize=3, label=label)
        build = lr.median([r["build_result"]["seconds"] for r in run["reps"]])
        axes[1].scatter(build, run["reps"][0]["build_result"]["index_bytes"] / 2**20)
        axes[1].annotate(label, (build, run["reps"][0]["build_result"]["index_bytes"] / 2**20),
                         fontsize=7)
    axes[0].set_xlim(right=1.002)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("recall@10")
    axes[0].set_ylabel("queries / s (1 thread)")
    axes[0].set_title("HNSW 100K x 128-d: recall vs throughput")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(True, which="both", alpha=0.3)
    axes[1].set_xlabel("serial build time (s)")
    axes[1].set_ylabel("graph memory (MiB)")
    axes[1].set_title("Build cost per configuration")
    axes[1].grid(True, alpha=0.3)
    save(fig, out, "hnsw_sweep_pareto.png")


def plot_exact(data, out):
    runs = lr.by_suite(data, "exact")
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    series = {}
    for run in runs.values():
        res = run["reps"][0]
        n, dim, simd = res["dataset"]["n"], res["dataset"]["dim"], res["build"]["simd_tier"]
        for t in res["results"]:
            series.setdefault((n, simd, t["threads"]), []).append((dim, t["qps_median"]))
    for (n, simd, threads), pts in sorted(series.items()):
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                linestyle="-" if simd == "avx2" else "--",
                label=f"N={n:,} {simd} {threads}T")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("dimension")
    ax.set_ylabel("queries / s")
    ax.set_title("Exact (Flat) search throughput, k = 10")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    save(fig, out, "exact_throughput.png")


def plot_threads(data, out):
    runs = lr.by_suite(data, "threads")
    fig, ax = plt.subplots(figsize=(7, 5))
    drawn = False
    for run_id, run in sorted(runs.items()):
        if run_id.startswith("threads-search"):
            per_t = {}
            for rep in run["reps"]:
                for r in rep["results"]:
                    per_t.setdefault(r["threads"], []).append(r["qps_median"])
            ts = sorted(per_t)
            base = lr.median(per_t[ts[0]])
            ax.plot(ts, [lr.median(per_t[t]) / base for t in ts], marker="o",
                    label=run_id.replace("threads-", ""))
            drawn = True
        elif run_id.startswith("threads-build"):
            per_t = {}
            for rep in run["reps"]:
                for r in rep["results"]:
                    if r["concurrency"] == "concurrent":
                        per_t.setdefault(r["threads"], []).append(r["build_seconds"])
            ts = sorted(per_t)
            base = lr.median(per_t[ts[0]])
            ax.plot(ts, [base / lr.median(per_t[t]) for t in ts], marker="s",
                    label=run_id.replace("threads-", ""))
            drawn = True
    if not drawn:
        plt.close(fig)
        return
    ax.plot([1, 16], [1, 16], color="grey", linestyle=":", label="linear")
    ax.axvline(8, color="grey", alpha=0.3)
    ax.set_xlabel("threads (8 physical cores)")
    ax.set_ylabel("speedup vs 1 thread")
    ax.set_title("Thread scaling")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    save(fig, out, "thread_scaling.png")


def plot_scale(data, out):
    runs = lr.by_suite(data, "scale")
    if not runs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    series = {}
    for run in runs.values():
        ds = run["reps"][0]["dataset"]
        r = run["reps"][0]["results"][0]
        at95 = next((p for p in r["recall"] if p["recall"] >= 0.95), None)
        series.setdefault(ds["dim"], []).append((ds["n"], r["build_seconds"],
                                                     at95["ef_search"] if at95 else None))
    for dim, pts in sorted(series.items()):
        pts.sort()
        axes[0].plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=f"d={dim}")
        axes[1].plot([p[0] for p in pts], [p[2] if p[2] else float("nan") for p in pts],
                     marker="o", label=f"d={dim}")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("N")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    axes[0].set_yscale("log")
    axes[0].set_ylabel("build time, 16 threads (s)")
    axes[0].set_title("HNSW build time (M=16, efC=200)")
    axes[1].set_yscale("log", base=2)
    axes[1].set_ylabel("smallest ef_search with recall@10 >= 0.95")
    axes[1].set_title("Search effort needed for recall 0.95")
    save(fig, out, "scale.png")


def plot_ingest(data, out):
    runs = lr.by_suite(data, "ingest")
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    labels, p50, p99, maxs = [], [], [], []
    for run_id, run in sorted(runs.items()):
        busy = [r["result"]["search_during_ingest"] for r in run["reps"]]
        labels.append(run_id.replace("ingest-hnsw-100k-d128-", ""))
        p50.append(lr.median([b["p50_us"] for b in busy]) / 1000)
        p99.append(lr.median([b["p999_us"] for b in busy]) / 1000)
        maxs.append(lr.median([b["max_us"] for b in busy]) / 1000)
    xs = range(len(labels))
    ax.bar([x - 0.25 for x in xs], p50, width=0.25, label="p50")
    ax.bar(list(xs), p99, width=0.25, label="p99.9")
    ax.bar([x + 0.25 for x in xs], maxs, width=0.25, label="max")
    ax.set_xticks(list(xs), labels)
    ax.set_yscale("log")
    ax.set_ylabel("search latency during ingestion (ms)")
    ax.set_title("HNSW 100K: searches while add_batch runs")
    ax.legend()
    ax.grid(True, axis="y", which="both", alpha=0.3)
    save(fig, out, "ingest_latency.png")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    run_dir = pathlib.Path(sys.argv[1])
    data = lr.load_run_dir(run_dir)
    out = run_dir / "plots"
    out.mkdir(exist_ok=True)
    for fn in (plot_sweep, plot_exact, plot_threads, plot_scale, plot_ingest):
        fn(data, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
