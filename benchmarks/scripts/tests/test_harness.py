"""Unit tests of the benchmark harness helpers (docs/benchmarking.md)."""

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import load_results as lr  # noqa: E402
import vfbench_lib as lib  # noqa: E402

CONFIGS = pathlib.Path(__file__).resolve().parents[2] / "configs"


def test_percentile_is_nearest_rank():
    samples = list(range(1, 101))  # 1..100
    assert lib.percentile(samples, 50) == 50
    assert lib.percentile(samples, 99) == 99
    assert lib.percentile(samples, 99.9) == 100
    assert lib.percentile(samples, 0) == 1
    assert lib.percentile(samples, 100) == 100
    assert lib.percentile([7.0], 99) == 7.0
    # Same definition as vf_bench: ceil(p/100 * n)-th smallest.
    assert lib.percentile([1, 2, 3, 4], 25) == 1
    assert lib.percentile([1, 2, 3, 4], 26) == 2
    with pytest.raises(ValueError):
        lib.percentile([], 50)
    with pytest.raises(ValueError):
        lib.percentile([1], 101)


def test_summarize_uses_the_upper_median():
    assert lib.summarize([3, 1, 2]) == {"median": 2, "min": 1, "max": 3, "n": 3}
    assert lib.summarize([4, 1, 3, 2])["median"] == 3
    with pytest.raises(ValueError):
        lib.summarize([])


def test_recall_at_k():
    exact = [[1, 2, 3], [4, 5, 6]]
    assert lib.recall_at_k(exact, exact, 3) == 1.0
    assert lib.recall_at_k([[1, 2, 9], [9, 9, 9]], exact, 3) == pytest.approx(1 / 3)
    assert lib.recall_at_k([[3, 2, 1], [6, 5, 4]], exact, 2) == 0.5
    assert lib.recall_at_k([], [], 10) == 0.0
    with pytest.raises(ValueError):
        lib.recall_at_k(exact, exact[:1], 3)
    with pytest.raises(ValueError):
        lib.recall_at_k(exact, exact, 0)


def test_first_ef_reaching_and_pareto():
    sweep = [{"ef_search": 64, "recall_at_k": 0.99, "qps": 10},
             {"ef_search": 16, "recall_at_k": 0.82, "qps": 25},
             {"ef_search": 32, "recall_at_k": 0.94, "qps": 16}]
    assert lib.first_ef_reaching(sweep, 0.90)["ef_search"] == 32
    assert lib.first_ef_reaching(sweep, 0.99)["ef_search"] == 64
    assert lib.first_ef_reaching(sweep, 0.999) is None
    front = lib.pareto([(0.9, 100), (0.95, 50), (0.9, 80), (0.99, 60), (0.8, 90)])
    assert front == [(0.9, 100), (0.99, 60)]


def test_expand_runs_matrix_exclude_and_placeholders():
    config = {
        "repeat": 2,
        "runs": [
            {"id": "r-{n}-{d}", "suite": "s", "tool": "vf_bench",
             "matrix": {"n": [1, 2], "d": [8, 16]}, "exclude": [{"n": 2, "d": 16}],
             "env": {"X": "{d}"}, "args": ["--n", "{n}", "--out", "{out}", "--f", "{work}/x"]},
            {"id": "plain", "suite": "s", "tool": "vf_bench", "repeat": 1, "args": []},
        ],
    }
    runs = lib.expand_runs(config)
    assert [r["id"] for r in runs] == ["r-1-8", "r-1-16", "r-2-8", "plain"]
    assert runs[0]["args"] == ["--n", "1", "--out", "{out}", "--f", "{work}/x"]
    assert runs[1]["env"] == {"X": "16"}
    assert runs[0]["repeat"] == 2 and runs[3]["repeat"] == 1
    assert runs[2]["bindings"] == {"n": 2, "d": 8}
    config["runs"].append({"id": "plain", "suite": "s", "tool": "vf_bench", "args": []})
    with pytest.raises(ValueError):
        lib.expand_runs(config)


def test_redaction_and_validation():
    gbench = {"context": {"host_name": "box", "executable": "C:/x.exe", "num_cpus": 16},
              "benchmarks": [{"name": "a"}]}
    assert lib.validate_result(gbench) == ["context.host_name is not redacted",
                                           "context.executable is not redacted"]
    redacted = lib.redact(gbench)
    assert redacted["context"]["host_name"] == lib.REDACTED
    assert redacted["context"]["num_cpus"] == 16
    assert lib.validate_result(redacted) == []

    good = {"schema": 1, "suite": "exact", "git": {"sha": "abc"}, "machine": {}, "build": {}}
    assert lib.validate_result(good) == []
    assert "schema must be 1" in lib.validate_result({**good, "schema": 2})
    assert "git.sha missing" in lib.validate_result({**good, "git": {}})
    assert lib.validate_result([]) == ["result is not a JSON object"]


def test_machine_info_has_no_host_or_user_names():
    import getpass
    import platform

    info = json.dumps(lib.machine_info())
    assert platform.node() not in info or platform.node() == ""
    assert getpass.getuser() not in info


@pytest.mark.parametrize("name", ["smoke.json", "full.json"])
def test_shipped_configs_expand(name):
    config = json.loads((CONFIGS / name).read_text(encoding="utf-8"))
    runs = lib.expand_runs(config)
    assert runs
    for run in runs:
        assert run["tool"] in ("vf_bench", "vf_micro_bench", "vf_http_bench", "vectorforge")
        assert "{" not in "".join(run["args"]).replace("{out}", "").replace("{work}", "")
    for entry in config.get("not_run", []):
        assert entry["reason"]


def test_load_run_dir_and_merged_sweep(tmp_path):
    rep = {"schema": 1, "suite": "hnsw-sweep", "git": {"sha": "x"}, "machine": {}, "build": {},
           "build_result": {"seconds": 1.0, "index_bytes": 10},
           "sweep": [{"ef_search": 10, "recall_at_k": 0.8, "qps": 100,
                      "latency_us": {"p50": 1, "p99": 2}, "dist_comps_mean": 5}]}
    (tmp_path / "a").mkdir()
    for i, qps in enumerate([100, 300, 200], start=1):
        rep["sweep"][0]["qps"] = qps
        (tmp_path / "a" / f"rep{i}.json").write_text(json.dumps(rep))
    manifest = {"runs": {"a": {"suite": "hnsw-sweep", "bindings": {},
                               "reps": [{"rep": 1, "status": "ok"}, {"rep": 2, "status": "ok"},
                                        {"rep": 3, "status": "failed"}]}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    data = lr.load_run_dir(tmp_path)
    assert len(data["runs"]["a"]["reps"]) == 2  # the failed repetition is ignored
    merged = lr.merged_sweep(data["runs"]["a"]["reps"])
    assert merged[0]["qps"] == 300 and merged[0]["qps_min"] == 100
    assert list(lr.by_suite(data, "hnsw-sweep")) == ["a"]
