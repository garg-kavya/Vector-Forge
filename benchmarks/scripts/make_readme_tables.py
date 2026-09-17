#!/usr/bin/env python3
"""Generates Markdown tables from a run directory (docs/benchmarking.md).

    python benchmarks/scripts/make_readme_tables.py benchmarks/results/<run> [--readme README.md]

Writes <run>/tables.md. With --readme, also replaces the text between
<!-- BENCHMARKS:BEGIN --> and <!-- BENCHMARKS:END --> in that file with the summary tables and a
link to the run directory. Every number comes from the run's result files.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import load_results as lr  # noqa: E402
import vfbench_lib as lib  # noqa: E402

BEGIN = "<!-- BENCHMARKS:BEGIN -->"
END = "<!-- BENCHMARKS:END -->"


def fmt(value, digits=0):
    if value is None:
        return "—"
    if isinstance(value, float) and digits:
        return f"{value:,.{digits}f}".replace(",", " ")
    return f"{value:,.0f}".replace(",", " ")


def table(header, rows):
    if not rows:
        return ""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def sweep_table(data):
    rows = []
    for run_id, run in sorted(lr.by_suite(data, "hnsw-sweep").items(),
                              key=lambda kv: (kv[1]["reps"][0]["params"]["M"],
                                              kv[1]["reps"][0]["params"]["ef_construction"],
                                              kv[1]["reps"][0]["params"]["k"])):
        prm = run["reps"][0]["params"]
        points = lr.merged_sweep(run["reps"])
        cells = []
        for target in (0.90, 0.95, 0.99):
            p = lib.first_ef_reaching(points, target)
            cells.append("—" if p is None else f"{fmt(p['qps'])} (ef {p['ef_search']})")
        build = lr.median([r["build_result"]["seconds"] for r in run["reps"]])
        mem = run["reps"][0]["build_result"]["index_bytes"] / 2**20
        rows.append([prm["M"], prm["ef_construction"], prm["k"], f"{build:.1f}", f"{mem:.0f}",
                     *cells])
    return table(["M", "efC", "k", "build s (1 thread)", "graph MiB",
                  "QPS @ recall≥0.90", "QPS @ ≥0.95", "QPS @ ≥0.99"], rows)


def exact_table(data):
    cells = {}
    for run in lr.by_suite(data, "exact").values():
        res = run["reps"][0]
        n, dim, simd = res["dataset"]["n"], res["dataset"]["dim"], res["build"]["simd_tier"]
        for t in res["results"]:
            cells[(n, dim, simd, t["threads"])] = t["qps_median"]
    dims = sorted({k[1] for k in cells})
    rows = []
    for n in sorted({k[0] for k in cells}):
        for simd in ("scalar", "avx2"):
            for threads in sorted({k[3] for k in cells}):
                row = [f"{n:,}".replace(",", " "), simd, threads]
                row += [fmt(cells.get((n, d, simd, threads)), 1) for d in dims]
                rows.append(row)
    return table(["N", "tier", "threads", *[f"d={d}" for d in dims]], rows)


def scale_table(data):
    rows = []
    for run in sorted(lr.by_suite(data, "scale").values(),
                      key=lambda r: (r["reps"][0]["dataset"]["n"], r["reps"][0]["dataset"]["dim"])):
        b = run["reps"][0]["dataset"]
        r = run["reps"][0]["results"][0]
        recall = {p["ef_search"]: p["recall"] for p in r["recall"]}
        at95 = next((ef for ef in sorted(recall) if recall[ef] >= 0.95), None)
        rows.append([f"{b['n']:,}".replace(",", " "), b["dim"], f"{r['build_seconds']:.1f}",
                     f"{r['index_bytes'] / 2**20:.0f}", f"{r['peak_rss_bytes'] / 2**30:.2f}",
                     f"{recall.get(64, float('nan')):.4f}", at95 or "> 512",
                     r["unreachable_level0"]])
    return table(["N", "d", "build s (16 threads)", "graph MiB", "peak RSS GiB",
                  "recall@10 ef=64", "ef for ≥0.95", "unreachable"], rows)


def threads_table(data):
    speedups = {}
    for run_id, run in sorted(lr.by_suite(data, "threads").items()):
        per_t = {}
        for rep in run["reps"]:
            for r in rep["results"]:
                if "qps_median" in r:
                    per_t.setdefault(r["threads"], []).append(r["qps_median"])
                elif r.get("concurrency") == "concurrent":
                    per_t.setdefault(r["threads"], []).append(1.0 / r["build_seconds"])
        base = lr.median(per_t[min(per_t)])
        speedups[run_id] = {t: lr.median(v) / base for t, v in per_t.items()}
    if not speedups:
        return ""
    columns = sorted({t for s in speedups.values() for t in s})
    rows = [[run_id.replace("threads-", ""),
             *[f"{s[t]:.2f}×" if t in s else "—" for t in columns]]
            for run_id, s in speedups.items()]
    return table(["workload", *[f"{t} T" for t in columns]], rows)


def ingest_table(data):
    rows = []
    for run_id, run in sorted(lr.by_suite(data, "ingest").items()):
        res = [r["result"] for r in run["reps"]]
        busy = [r["search_during_ingest"] for r in res]
        rows.append([run_id.replace("ingest-hnsw-100k-d128-", ""),
                     fmt(lr.median([r["ingest_vectors_per_s"] for r in res])),
                     f"{lr.median([r['search_idle']['p50_us'] for r in res]) / 1000:.3f}",
                     f"{lr.median([b['p50_us'] for b in busy]) / 1000:.3f}",
                     f"{lr.median([b['p999_us'] for b in busy]) / 1000:.2f}",
                     f"{lr.median([b['max_us'] for b in busy]) / 1000:.2f}"])
    return table(["mode / writer threads", "vectors/s", "idle p50 ms", "busy p50 ms",
                  "busy p99.9 ms", "busy max ms"], rows)


def storage_tables(data):
    load_rows, save_rows = [], []
    for run_id, run in sorted(lr.by_suite(data, "storage").items()):
        reps = run["reps"]
        results = [r["result"] for r in reps]
        if "save_seconds" in results[0]:
            save_rows.append([run_id, f"{lr.median([r['build_seconds'] for r in results]):.1f}",
                              str(reps[0]["params"]["threads"]),
                              f"{lr.median([r['save_seconds'] for r in results]):.2f}",
                              f"{results[0]['file_bytes'] / 2**20:.0f}"])
        else:
            open_ms = lr.median([r["open_seconds"] for r in results]) * 1000
            first = lr.median([r["first_query_us"] for r in results]) / 1000
            steady = lr.median([r["pass2"]["qps"] for r in results])
            rss = lr.median([r["rss_bytes"]["after_open"] - r["rss_bytes"]["before"]
                             for r in results]) / 2**20
            load_rows.append([run_id, f"{open_ms:.0f}", f"{first:.2f}", fmt(steady), f"+{rss:.0f}"])
    return (table(["run", "open ms", "first query ms", "steady QPS", "RSS after open MiB"], load_rows),
            table(["run", "build s", "build threads", "save s", "file MiB"], save_rows))


def not_run_table(data):
    return table(["scenario", "reason"],
                 [[r["id"], r["reason"]] for r in data["manifest"].get("not_run", [])])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--readme")
    args = parser.parse_args()
    run_dir = pathlib.Path(args.run_dir)
    data = lr.load_run_dir(run_dir)
    load_table, save_table = storage_tables(data)
    sections = [
        ("Exact search (queries/s, k = 10)", exact_table(data)),
        ("HNSW parameter sweep (100K × 128-d, one thread)", sweep_table(data)),
        ("HNSW scale (M = 16, efC = 200, 16 build threads)", scale_table(data)),
        ("Thread scaling (speedup vs one thread)", threads_table(data)),
        ("Search latency during ingestion", ingest_table(data)),
        ("Loading (warm file cache)", load_table),
        ("Saving", save_table),
        ("Not run", not_run_table(data)),
    ]
    body = "\n\n".join(f"### {title}\n\n{text}" for title, text in sections if text)
    (run_dir / "tables.md").write_text(body + "\n", encoding="utf-8", newline="\n")
    print("wrote", run_dir / "tables.md")
    if args.readme:
        readme = pathlib.Path(args.readme)
        text = readme.read_text(encoding="utf-8")
        if BEGIN not in text or END not in text:
            print(f"{readme}: markers not found", file=sys.stderr)
            return 1
        rel = run_dir.as_posix()
        summary = "\n\n".join(f"#### {t}\n\n{s}" for t, s in sections[:5] if s)
        block = (f"{BEGIN}\n<!-- generated by benchmarks/scripts/make_readme_tables.py from {rel} -->\n\n"
                 f"{summary}\n\nRaw data, environment and interpretation: [{rel}]({rel}/README.md).\n{END}")
        text = re.sub(re.escape(BEGIN) + ".*?" + re.escape(END), lambda _m: block, text,
                      flags=re.S)
        readme.write_text(text, encoding="utf-8", newline="\n")
        print("updated", readme)
    return 0


if __name__ == "__main__":
    sys.exit(main())
