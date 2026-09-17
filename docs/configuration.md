# Configuration

Every setting a user can change, where it is set, and its default.

## Collections (`vf::CollectionConfig`, `include/vectorforge/config.hpp`)

| Field | Default | Meaning |
|---|---|---|
| `dim` | required | vector dimension, 1 … 65 536 |
| `metric` | `L2` | `L2` (squared distance), `InnerProduct` (distance −⟨a, b⟩), `Cosine` (1 − cos) |
| `normalize` | false | normalise stored and query vectors (always on for cosine) |
| `index` | `Hnsw` | `Hnsw` (approximate) or `Flat` (exact) |
| `hnsw.M` | 16 | links per node on upper levels (2·M on level 0), 2 … 512 |
| `hnsw.ef_construction` | 200 | build beam width, ≥ M |
| `hnsw.ef_search` | 50 | default query beam width; `SearchParams::ef_search` overrides per query |
| `hnsw.max_level` | 16 | cap on node levels |
| `hnsw.seed` | 0x5EEDF0A6E | level assignment seed (builds are deterministic for a fixed seed and insertion order) |
| `concurrency` | `Concurrent` | HNSW insert synchronisation: `Concurrent` (Level B) or `Coarse` (Level A); not stored in files |

Stored in index files and `config.json`: everything except `concurrency`. Guidance on M, ef and
their costs: [hnsw.md](hnsw.md) and the parameter sweep in the benchmark results.

## Loading (`vf::LoadOptions`)

| Field | Default | Meaning |
|---|---|---|
| `use_mmap` | true | serve vectors from a read-only mapping (small files are copied anyway) |
| `verify` | `Auto` | checksum verification: `Auto` (Full for heap, Metadata for mmap), `None`, `Metadata`, `Full` |
| `prefault` | false | page the mapped vectors in at load |
| `concurrency` | `Concurrent` | `CollectionConfig::concurrency` of the loaded collection |

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `VF_SIMD` | `auto` | kernel tier: `auto`, `scalar`, `avx2`; an unsupported or invalid value is an error for every API call and command ([simd.md](simd.md)) |
| `VF_API_KEY` | unset | bearer token for `vectorforge serve` when `--api-key` is not given |
| `VF_DATA_DIR` | `datasets` | download directory of `tools/datasets/fetch.py` |

## `vectorforge serve`

| Option | Default | Meaning |
|---|---|---|
| `--data-dir` | required | catalog directory |
| `--host`, `--port` | `127.0.0.1`, 8080 | listen address (`--port 0`: any free port) |
| `--http-threads` | hardware threads | connection threads |
| `--threads` | hardware threads | compute pool for batch requests |
| `--api-key` | `$VF_API_KEY` | require `Authorization: Bearer …` on `/v1` routes and `/metrics` |
| `--max-body-mb` | 64 | request body limit |
| `--max-batch` | 10 000 | vectors per insert or search batch |
| `--max-k`, `--max-ef`, `--max-dim` | 1 000, 4 096, 65 536 | request caps |
| `--no-mmap` | off | load snapshots into memory |
| `--snapshot-on-exit` | off | snapshot every collection after a graceful shutdown |
| `--drain-seconds` | 10 | how long shutdown waits for requests in progress |
| `--log-level` | `info` | `debug`, `info`, `warn`, `error`, `off` |
| `--access-log` | off | one log line per request |
| `--list-routes` | — | print the route table and exit |

Fixed server settings (`vf::server::ServerConfig`, for embedding): read/write timeouts 30 s,
keep-alive timeout 5 s, at most 4 batch requests on the compute pool at once.

## Logging

`vectorforge serve` writes structured lines to stderr in logfmt:

```text
ts=2026-09-17T10:00:00.123Z level=info event=server_started host=127.0.0.1 port=8080 collections=2 auth=false
ts=2026-09-17T10:00:01.002Z level=info event=request method=POST path=/v1/collections/docs/search status=200 ms=0.412 id=6f1c…-17
```

Values with spaces, quotes, `=` or control characters are quoted and escaped, so request data
cannot forge or split lines. Events: `server_started`, `server_stopping`, `snapshots_written`,
`snapshot_failed`, `request` (with `--access-log`). Human-readable status lines also go to stdout.

## Metrics

`GET /metrics` (Prometheus text format; requires the bearer token when one is configured):

| Metric | Type | Labels |
|---|---|---|
| `vectorforge_http_requests_total` | counter | `method`, `route` (OpenAPI path or `unmatched`), `status` (`2xx` …) |
| `vectorforge_http_request_duration_seconds` | histogram (50 µs … 10 s) | `method`, `route` |
| `vectorforge_http_in_flight_requests` | gauge | — |
| `vectorforge_collections` | gauge | — |
| `vectorforge_collection_vectors`, `…_deleted_vectors`, `…_memory_bytes`, `…_snapshot_generation` | gauge | `collection` |
| `vectorforge_uptime_seconds` | gauge | — |
| `vectorforge_build_info` | gauge (1) | `version`, `git_sha`, `simd` |

## Build options

See the README ("Useful CMake options") and `CMakePresets.json`. The presets enable warnings as
errors, tests, benchmarks, the CLI and the server.
