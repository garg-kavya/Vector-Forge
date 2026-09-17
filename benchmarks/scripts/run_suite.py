#!/usr/bin/env python3
"""Runs a benchmark configuration, one fresh process per repetition (docs/benchmarking.md).

    python benchmarks/scripts/run_suite.py benchmarks/configs/full.json \\
        --build-dir out/build/msvc-release --out benchmarks/results/<date>_<machine>_<build>_suite

Layout of --out:
    manifest.json            machine, build, configuration, and the status of every run
    <run id>/rep<N>.json     tool output of repetition N (redacted), or
    <run id>/rep<N>.log      stderr of a failed repetition

Placeholders in a run's arguments: {out} (the repetition's result file), {work} (a scratch
directory for index files, --work, default out/bench-work). Runs execute in file order, so a run
may use files written by an earlier one.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import vfbench_lib as lib  # noqa: E402

TOOLS = {
    "vf_bench": "benchmarks/vf_bench",
    "vf_micro_bench": "benchmarks/vf_micro_bench",
    "vf_http_bench": "benchmarks/vf_http_bench",
    "vectorforge": "apps/cli/vectorforge",
}


def tool_path(build_dir: pathlib.Path, tool: str) -> pathlib.Path:
    rel = TOOLS[tool]
    for suffix in (".exe", ""):
        candidate = build_dir / (rel + suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{tool} not found under {build_dir}")


def git_state(root: pathlib.Path) -> dict:
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                              text=True).stdout.strip()
    sha = git("rev-parse", "--short=12", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    return {"sha": sha, "dirty": dirty}


def load_json(path: pathlib.Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: pathlib.Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=1)
        f.write("\n")


def run_one(run: dict, rep: int, build_dir: pathlib.Path, out_dir: pathlib.Path,
            work_dir: pathlib.Path) -> dict:
    target = out_dir / run["id"] / f"rep{rep}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    tool = tool_path(build_dir, run["tool"])
    args = [a.replace("{out}", str(target)).replace("{work}", str(work_dir)) for a in run["args"]]
    env = dict(os.environ)
    env.update({k: str(v) for k, v in run.get("env", {}).items()})
    start = time.perf_counter()
    proc = subprocess.run([str(tool), *args], capture_output=True, text=True, env=env,
                          timeout=run.get("timeout_s", 3600))
    seconds = time.perf_counter() - start
    status = {"rep": rep, "seconds": round(seconds, 3), "returncode": proc.returncode}
    if proc.returncode != 0:
        (target.with_suffix(".log")).write_text(proc.stdout[-5000:] + proc.stderr[-20000:],
                                                encoding="utf-8")
        status["status"] = "failed"
        return status
    if not target.exists():  # tools that only print JSON
        target.write_text(proc.stdout, encoding="utf-8")
    result = lib.redact(load_json(target))
    write_json(target, result)
    problems = lib.validate_result(result)
    status["status"] = "ok" if not problems else "invalid"
    if problems:
        status["problems"] = problems
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--build-dir")
    parser.add_argument("--out")
    parser.add_argument("--work", default="out/bench-work")
    parser.add_argument("--only", help="regular expression on run ids")
    parser.add_argument("--list", action="store_true", help="print the runs and exit")
    parser.add_argument("--resume", action="store_true",
                        help="skip repetitions whose result file already exists")
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parents[2]
    config = load_json(pathlib.Path(args.config))
    runs = lib.expand_runs(config)
    if args.only:
        runs = [r for r in runs if re.search(args.only, r["id"])]
    if args.list:
        for r in runs:
            print(f"{r['id']:48} x{r['repeat']}  {r['tool']} {' '.join(r['args'])}")
        for r in config.get("not_run", []):
            print(f"{r['id']:48} not run: {r['reason']}")
        return 0

    if not args.build_dir or not args.out:
        parser.error("--build-dir and --out are required unless --list is given")
    build_dir = pathlib.Path(args.build_dir)
    out_dir = pathlib.Path(args.out)
    work_dir = pathlib.Path(args.work).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = load_json(manifest_path) if (args.resume and manifest_path.exists()) else {}
    manifest.update({
        "schema": lib.SCHEMA,
        "suite": config.get("suite"),
        "description": config.get("description", ""),
        "started": manifest.get("started", dt.datetime.now(dt.timezone.utc).isoformat()),
        "git": git_state(root),
        "machine": lib.machine_info(),
        "build_dir_name": build_dir.name,
        "config": config,
        "not_run": config.get("not_run", []),
    })
    manifest.setdefault("runs", {})
    failures = 0
    for run in runs:
        entry = manifest["runs"].setdefault(run["id"], {"suite": run["suite"], "reps": []})
        entry.update({"suite": run["suite"], "tool": run["tool"], "args": run["args"],
                      "env": run.get("env", {}), "bindings": run["bindings"]})
        done = {r["rep"] for r in entry["reps"] if r["status"] == "ok"}
        for rep in range(1, run["repeat"] + 1):
            if args.resume and rep in done and (out_dir / run["id"] / f"rep{rep}.json").exists():
                continue
            print(f"[{dt.datetime.now():%H:%M:%S}] {run['id']} rep {rep}/{run['repeat']}",
                  flush=True)
            status = run_one(run, rep, build_dir, out_dir, work_dir)
            entry["reps"] = [r for r in entry["reps"] if r["rep"] != rep] + [status]
            if status["status"] != "ok":
                failures += 1
                print(f"    {status}", flush=True)
            write_json(manifest_path, manifest)
    manifest["finished"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(manifest_path, manifest)
    if not args.keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)
    print(f"done: {len(runs)} runs, {failures} failed repetitions")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
