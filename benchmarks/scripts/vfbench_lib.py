"""Shared helpers of the benchmark harness (docs/benchmarking.md).

Pure functions only (no process launching), so they are unit-tested in
benchmarks/scripts/tests/test_harness.py.
"""

from __future__ import annotations

import copy
import itertools
import math
import os
import platform
import re
import subprocess
from typing import Any, Iterable

SCHEMA = 1
REDACTED = "<redacted>"
# Keys whose values identify the machine or the local file system (Google Benchmark context).
PRIVATE_KEYS = {"host_name", "executable"}


def percentile(sorted_samples: list[float], p: float) -> float:
    """Nearest-rank percentile of already sorted samples (the definition vf_bench uses)."""
    if not sorted_samples:
        raise ValueError("no samples")
    if not 0.0 <= p <= 100.0:
        raise ValueError("p must be in [0, 100]")
    rank = math.ceil(p / 100.0 * len(sorted_samples))
    return sorted_samples[min(len(sorted_samples) - 1, max(rank, 1) - 1)]


def summarize(values: Iterable[float]) -> dict[str, float]:
    """Median (upper median for even counts, like vf_bench), min and max."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no values")
    return {"median": ordered[len(ordered) // 2], "min": ordered[0], "max": ordered[-1],
            "n": len(ordered)}


def recall_at_k(approx: list[list[int]], exact: list[list[int]], k: int) -> float:
    """Mean id-overlap recall@k (ties are not special-cased; vf_bench uses the tie-tolerant C++
    definition, which is never lower)."""
    if k <= 0 or len(approx) != len(exact):
        raise ValueError("bad recall arguments")
    if not approx:
        return 0.0
    total = 0.0
    for a, e in zip(approx, exact):
        total += len(set(a[:k]) & set(e[:k])) / k
    return total / len(approx)


def first_ef_reaching(sweep: list[dict[str, Any]], target: float) -> dict[str, Any] | None:
    """The first sweep point (in ascending ef order) whose recall reaches `target`: the
    interpolation-free recall target of docs/DESIGN.md §16.4."""
    for point in sorted(sweep, key=lambda p: p["ef_search"]):
        if point["recall_at_k"] >= target:
            return point
    return None


def pareto(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Points (recall, qps) not dominated by another point, sorted by recall."""
    front = []
    for r, q in sorted(points, key=lambda p: (-p[0], -p[1])):
        if not front or q > front[-1][1]:
            front.append((r, q))
    return sorted(front)


def _format(value: Any, bindings: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format(**bindings)
    if isinstance(value, list):
        return [_format(v, bindings) for v in value]
    if isinstance(value, dict):
        return {k: _format(v, bindings) for k, v in value.items()}
    return value


def expand_runs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expands `matrix` entries into concrete runs; `exclude` drops combinations. Placeholders in
    strings ({name}) are filled from the matrix and from `vars`."""
    runs = []
    for entry in config.get("runs", []):
        matrix = entry.get("matrix", {})
        keys = list(matrix)
        excludes = entry.get("exclude", [])
        for combo in itertools.product(*(matrix[k] for k in keys)) if keys else [()]:
            bindings = dict(config.get("vars", {}))
            bindings.update(dict(zip(keys, combo)))
            if any(all(bindings.get(k) == v for k, v in ex.items()) for ex in excludes):
                continue
            run = {k: copy.deepcopy(v) for k, v in entry.items() if k not in ("matrix", "exclude")}
            run = _format(run, {**bindings, "out": "{out}", "work": "{work}"})
            run.setdefault("repeat", config.get("repeat", 1))
            run.setdefault("env", {})
            run["bindings"] = bindings
            runs.append(run)
    ids = [r["id"] for r in runs]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate run ids: {sorted(duplicates)}")
    return runs


def redact(value: Any) -> Any:
    """Replaces machine-identifying values (host name, executable path) in a result."""
    if isinstance(value, dict):
        return {k: (REDACTED if k in PRIVATE_KEYS else redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


def validate_result(result: Any) -> list[str]:
    """Problems with one tool output (empty list: valid)."""
    problems = []
    if not isinstance(result, dict):
        return ["result is not a JSON object"]
    if "benchmarks" in result and "context" in result:  # Google Benchmark
        if not result["benchmarks"]:
            problems.append("no benchmarks")
        for key in PRIVATE_KEYS:
            if result["context"].get(key, REDACTED) != REDACTED:
                problems.append(f"context.{key} is not redacted")
        return problems
    if result.get("schema") != SCHEMA:
        problems.append("schema must be 1")
    if not isinstance(result.get("suite"), str):
        problems.append("suite missing")
    for section in ("git", "machine", "build"):
        if not isinstance(result.get(section), dict):
            problems.append(f"{section} missing")
    if isinstance(result.get("git"), dict) and not result["git"].get("sha"):
        problems.append("git.sha missing")
    return problems


def machine_info() -> dict[str, Any]:
    """Hardware and OS facts for the run manifest (no host name, user or paths)."""
    info: dict[str, Any] = {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "arch": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
    }
    try:
        import psutil  # optional

        info["physical_cpus"] = psutil.cpu_count(logical=False)
        info["ram_bytes"] = psutil.virtual_memory().total
        battery = psutil.sensors_battery()
        info["ac_power"] = None if battery is None else bool(battery.power_plugged)
    except ImportError:
        pass
    if platform.system() == "Windows":
        info.update(_windows_facts())
    elif platform.system() == "Linux":
        info.update(_linux_facts())
    return info


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _windows_facts() -> dict[str, Any]:
    facts: dict[str, Any] = {}
    scheme = _run(["powercfg", "/getactivescheme"])
    m = re.search(r"\(([^)]+)\)\s*$", scheme.strip())
    if m:
        facts["power_plan"] = m.group(1)
    ps = _run(["powershell", "-NoProfile", "-Command",
               "$c=Get-CimInstance Win32_Processor; $m=Get-CimInstance Win32_ComputerSystem;"
               "\"$($c.NumberOfCores) $($m.TotalPhysicalMemory)\""])
    parts = ps.split()
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        facts.setdefault("physical_cpus", int(parts[0]))
        facts.setdefault("ram_bytes", int(parts[1]))
    return facts


def _linux_facts() -> dict[str, Any]:
    facts: dict[str, Any] = {}
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    facts["ram_bytes"] = int(line.split()[1]) * 1024
        with open("/proc/cpuinfo", encoding="ascii") as f:
            cores = {line for line in f if line.startswith("core id")}
            if cores:
                facts["physical_cpus"] = len(cores)
    except OSError:
        pass
    governor = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
    if os.path.exists(governor):
        with open(governor, encoding="ascii") as f:
            facts["cpufreq_governor"] = f.read().strip()
    return facts
