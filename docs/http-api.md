# HTTP API

`vectorforge serve` exposes a catalog of named collections over HTTP/JSON. The machine-readable
description is [openapi.yaml](openapi.yaml) (checked against the server's route table in CI by
`tools/check_openapi.py`); a runnable walkthrough is
[examples/http/curl_examples.sh](../examples/http/curl_examples.sh). Design:
[DESIGN.md §13](DESIGN.md#13-http-api-design).

Source: `src/server/` (server library `vf_server`: cpp-httplib 0.54.1 and nlohmann/json 3.12.0,
both pinned by SHA-256), `include/vectorforge/catalog.hpp`, `src/collection/catalog.cpp`,
`apps/cli/cmd_serve.cpp`.

## Running

```bash
vectorforge serve --data-dir ./data                      # 127.0.0.1:8080
vectorforge serve --data-dir ./data --host 0.0.0.0 --port 9000 --threads 8 --snapshot-on-exit
VF_API_KEY=secret vectorforge serve --data-dir ./data     # require a bearer token
vectorforge serve --list-routes                           # print the route table and exit
```

| Option | Default | Meaning |
|---|---|---|
| `--data-dir` | required | catalog directory (created if missing) |
| `--host`, `--port` | `127.0.0.1`, `8080` | listen address; `--port 0` picks a free port |
| `--http-threads` | hardware threads | connection handler threads |
| `--threads` | hardware threads | compute threads for batch inserts and batch searches |
| `--api-key` / `VF_API_KEY` | none | require `Authorization: Bearer <key>` on `/v1` routes (prefer the environment variable: command lines are visible to other users) |
| `--max-body-mb`, `--max-batch`, `--max-k`, `--max-ef`, `--max-dim` | 64, 10 000, 1 000, 4 096, 65 536 | request limits |
| `--no-mmap` | off | load snapshots into memory instead of mapping them |
| `--snapshot-on-exit` | off | snapshot every collection after a graceful shutdown |
| `--drain-seconds` | 10 | how long shutdown waits for requests in progress |

**Lifecycle.** On start the catalog loads every collection (from its newest snapshot, or empty
from `config.json` if it was never snapshotted). SIGINT/SIGTERM (Ctrl+C, console close or system
shutdown on Windows) starts a graceful shutdown: `/readyz` and all `/v1` routes answer 503, requests
already running finish (up to `--drain-seconds`), the listener closes, and with
`--snapshot-on-exit` every collection is snapshotted.

**Durability.** Inserts live in memory until `POST .../snapshot` (or `--snapshot-on-exit`). A
snapshot writes a new generation file and atomically switches `MANIFEST` to it
([storage-format.md](storage-format.md#snapshots)); a crash at any point leaves the previous
generation loadable.

## Conventions

- JSON bodies require `Content-Type: application/json` (415 otherwise); request objects are
  strict: unknown fields are errors, ids are unsigned integers (`0 … 2^64 − 2`), vector components
  are numbers. JSON nesting deeper than 8 levels is rejected before parsing.
- Every response carries `X-Request-Id`: the request's own header if it is 1–128 printable ASCII
  characters, otherwise a generated id.
- Search responses include `took_ms` (server-side handling time, excluding network transfer).
- Distances follow the library: squared L2, negative inner product, or cosine distance
  `1 − cos` (README, "Distance semantics").
- Errors: `{"error": {"code": "DIMENSION_MISMATCH", "message": "vector has 3 components, expected 4"}}`.

| Status | Codes | When |
|---|---|---|
| 400 | `INVALID_ARGUMENT`, `DIMENSION_MISMATCH`, `MALFORMED_JSON` | bad request content |
| 401 | `UNAUTHORIZED` | missing or wrong bearer token |
| 404 | `NOT_FOUND` | unknown collection, vector id or route |
| 409 | `ALREADY_EXISTS`, `FAILED_PRECONDITION` | id or collection exists |
| 413 | `PAYLOAD_TOO_LARGE` | body over `--max-body-mb` |
| 415 | `UNSUPPORTED_MEDIA_TYPE` | wrong `Content-Type` |
| 422 | `LIMIT_EXCEEDED` | batch, `k`, `ef_search` or `dim` above the configured limit |
| 500 | `CORRUPT_DATA`, `UNSUPPORTED_VERSION`, `IO_ERROR`, `INTERNAL` | server-side failure |
| 503 | `UNAVAILABLE` | shutting down, or a dropped collection's name is still in use |
| 507 | `RESOURCE_EXHAUSTED` | out of memory or collection capacity |

## Endpoints

| Method & path | Body | Response |
|---|---|---|
| `GET /healthz` | — | `{"status": "ok"}` (no auth) |
| `GET /readyz` | — | 200 `{"status": "ready"}` or 503 (no auth) |
| `GET /v1/status` | — | version, git SHA, build, SIMD tier, uptime, collection count, thread counts |
| `POST /v1/collections` | `{"name", "dim", "metric"?, "normalize"?, "index"?: {"type", "M", "ef_construction", "ef_search", "max_level", "seed", "concurrency"}}` | 201, the collection |
| `GET /v1/collections` | — | `{"collections": [...]}` in name order |
| `GET /v1/collections/{name}` | — | `{"name", "dim", "metric", "normalize", "index", "count"}` |
| `DELETE /v1/collections/{name}` | — | `{"dropped": name}` |
| `POST /v1/collections/{name}/vectors` | `{"upsert"?: false, "vectors": [{"id", "vector"}]}` | `{"inserted": n}` |
| `POST /v1/collections/{name}/vectors:bulk[?upsert=true]` | binary (below) | `{"inserted": n}` |
| `GET /v1/collections/{name}/vectors/{id}` | — | `{"id", "vector"}` (normalised for normalised collections) |
| `DELETE /v1/collections/{name}/vectors/{id}` | — | `{"deleted": id}` |
| `POST /v1/collections/{name}/search` | `{"vector", "k"?: 10, "ef_search"?}` | `{"results": [{"id", "distance"}], "took_ms"}` |
| `POST /v1/collections/{name}/search:batch` | `{"vectors": [[...]], "k"?, "ef_search"?}` | `{"results": [[...], ...], "took_ms"}` |
| `GET /v1/collections/{name}/stats` | — | counts, deleted ratio, memory breakdown, SIMD tier |
| `POST /v1/collections/{name}/snapshot` | — | `{"generation", "bytes", "took_ms"}` |
| `POST /v1/collections/{name}/compact` | — | `{"rows_before", "removed_rows", "rows_after", "took_ms"}` |

Semantics are those of the library (`include/vectorforge/collection.hpp`): a batch is validated as
a whole before anything is inserted (existing ids fail with 409 unless `upsert`), searches run
concurrently with inserts, and a removed id is never returned by a search that starts after the
removal answered.

**Binary bulk format** (`Content-Type: application/octet-stream`), little-endian:

```text
"VFB1"  u32 dim  u64 count  | count x u64 id | count x dim x f32 |
```

The body length must be exactly `16 + 8·count + 4·dim·count`.

**Dropping.** `DELETE /v1/collections/{name}` removes the collection immediately; requests still
running keep using it, and its files are deleted when the last one finishes. Until then, creating
a collection with the same name answers 503. A `DROPPED` marker makes the drop survive a crash.

## Threads and limits

Each connection is served by one of `--http-threads` threads. Single searches and inserts run on
that thread. Batch inserts and batch searches additionally use a shared pool of `--threads`
compute threads; at most four requests use the pool at once, others run on their own thread, so
the number of busy threads stays bounded by `http_threads + compute_threads`.

JSON costs memory: nlohmann/json holds a parsed number in 16 bytes plus array overhead, so a
64 MB body of numbers can take several hundred MB while it is decoded. Use `vectors:bulk` for
large ingestion. Measured costs: see "Measurements".

## Security notes

- The server binds `127.0.0.1` by default. There is no TLS; put a TLS-terminating proxy in front
  when exposing it.
- The bearer token is compared in constant time; `/healthz` and `/readyz` are always open.
- Collection names are restricted to `[A-Za-z0-9_-]{1,64}` before they are used as directory names.
- Request bodies are limited in size, nesting depth, batch length, `k`, `ef_search` and `dim`.
- The Docker image runs as an unprivileged user.

## Docker

```bash
docker build -t vectorforge .
docker run --rm -p 127.0.0.1:8080:8080 -v vf-data:/data -e VF_API_KEY=secret vectorforge
docker compose up --build        # docker-compose.yml: volume, snapshot on exit, 60 s stop grace
```

The CI workflow `.github/workflows/docker.yml` builds the image, runs the curl walkthrough against
it, stops it with SIGTERM and checks that the collection is still there after a restart. (No
Docker engine was available on the development machine; the image is exercised only in CI.)

## Measurements

Ryzen 7 4800H, Windows 11, loopback, one keep-alive client, HNSW 100 000 × 128-d, k = 10,
ef_search = 64 ([results](../benchmarks/results/2026-09-17_ryzen7-4800h_msvc-release_phase7/README.md)):

| | p50 | p99 |
|---|---|---|
| `Collection::search` in-process | 0.11 ms | 0.19–0.23 ms |
| server handling (`took_ms`) | 0.15–0.16 ms | 0.39–0.40 ms |
| JSON decoding of the request alone (2.6 KB) | 0.06 ms | 0.14 ms |
| HTTP round trip at the client | 0.33–0.35 ms | 0.96–1.21 ms |

Ingestion of 100 000 × 128-d vectors in requests of 1 000: 18 000–18 700 vectors/s through JSON
(255 MB sent) against 406 000–464 000 vectors/s through `vectors:bulk` (52 MB).
