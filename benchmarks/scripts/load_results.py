"""Reads a run directory written by run_suite.py into per-run records (docs/benchmarking.md)."""

from __future__ import annotations

import json
import pathlib
from typing import Any


def load_run_dir(run_dir: str | pathlib.Path) -> dict[str, Any]:
    """{"manifest": ..., "runs": {id: {"meta": manifest entry, "reps": [result, ...]}}}.
    Only repetitions with status "ok" are loaded."""
    root = pathlib.Path(run_dir)
    with open(root / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    runs = {}
    for run_id, meta in manifest.get("runs", {}).items():
        reps = []
        for rep in sorted(meta.get("reps", []), key=lambda r: r["rep"]):
            path = root / run_id / f"rep{rep['rep']}.json"
            if rep.get("status") == "ok" and path.exists():
                with open(path, encoding="utf-8") as f:
                    reps.append(json.load(f))
        runs[run_id] = {"meta": meta, "reps": reps}
    return {"manifest": manifest, "runs": runs}


def by_suite(data: dict[str, Any], suite: str) -> dict[str, dict[str, Any]]:
    return {k: v for k, v in data["runs"].items() if v["meta"].get("suite") == suite and v["reps"]}


def median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def merged_sweep(reps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per ef_search: median QPS and latency across repetitions; recall of the first repetition
    (recall is deterministic for a fixed build)."""
    points = []
    for i, first in enumerate(reps[0]["sweep"]):
        qps = [r["sweep"][i]["qps"] for r in reps]
        p50 = [r["sweep"][i]["latency_us"]["p50"] for r in reps]
        p99 = [r["sweep"][i]["latency_us"]["p99"] for r in reps]
        points.append({
            "ef_search": first["ef_search"],
            "recall_at_k": first["recall_at_k"],
            "recall_spread": max(r["sweep"][i]["recall_at_k"] for r in reps)
            - min(r["sweep"][i]["recall_at_k"] for r in reps),
            "qps": median(qps), "qps_min": min(qps), "qps_max": max(qps),
            "p50_us": median(p50), "p99_us": median(p99),
            "dist_comps_mean": first.get("dist_comps_mean"),
        })
    return points
