# Phase 7 HTTP overhead — 2026-09-17

Raw data: [`http_hnsw_100k_d128.json`](http_hnsw_100k_d128.json) and
[`http_hnsw_100k_d128_run2.json`](http_hnsw_100k_d128_run2.json) (`vf_http_bench`, two runs of the
same configuration).

## Environment and protocol

Ryzen 7 4800H (8 cores / 16 threads), Windows 11, MSVC 19.50 `msvc-release`, AVX2 tier — the
machine of the [Phase 6a results](../2026-09-16_ryzen7-4800h_msvc-release_phase6a/README.md).
Source: `418517e1a2b1-dirty`, the working tree that became the Phase 7 commit.

- An in-process `vf::server::Server` (default thread counts) on 127.0.0.1 serves a catalog in a
  temporary directory; one `httplib::Client` thread sends requests over one keep-alive connection.
- Search: 100 000 × 128-d Gaussian-mixture vectors in an HNSW collection (M = 16,
  ef_construction = 200, built with a pool), k = 10, ef_search = 64. For each of 2 000 queries the
  benchmark times, in this order: `Collection::search` in-process; `decode_search` on the request
  body alone; the HTTP round trip. The server's own `took_ms` (routing, decoding, search,
  encoding) is read from each response. 200 warm-up queries first.
- Ingestion: 100 000 vectors into two Flat collections (so that transfer and decoding dominate), in
  requests of 1 000 vectors, through `POST .../vectors` (JSON) and `POST .../vectors:bulk`
  (binary). Client-side JSON encoding is timed separately and excluded from the rate.

## Results

| Single query (µs) | run 1 p50 | run 1 p99 | run 2 p50 | run 2 p99 |
|---|---|---|---|---|
| in-process `search` | 110 | 193 | 116 | 233 |
| server handling (`took_ms`) | 151 | 397 | 160 | 387 |
| of which request decoding (2.6 KB JSON) | 58 | 136 | 60 | 137 |
| HTTP round trip (client) | 332 | 1 210 | 351 | 957 |

| Ingestion, 100 000 × 128-d | run 1 | run 2 | bytes sent |
|---|---|---|---|
| JSON `vectors` | 18 047 vec/s | 18 739 vec/s | 255 MB |
| binary `vectors:bulk` | 406 266 vec/s | 463 624 vec/s | 52 MB |

## Interpretation

- A single HNSW query costs ~0.11 ms in-process and ~0.33–0.35 ms (median) through HTTP on
  loopback. The server-side handling adds ~0.04 ms beyond the search; most of that is JSON decoding
  of the 128 query components (~0.06 ms per request measured in isolation, which overlaps with the
  search time variation between the two measurements). The remaining ~0.18 ms is the client,
  the loopback TCP stack and the HTTP framing on Windows — not measured separately.
- The p99 round trip (~1 ms) is 3–5× its median; the server-side p99 is ~0.4 ms, so most of the
  tail is outside the handler (scheduling of the connection thread and the client). Not
  investigated further.
- JSON ingestion is ~22–25× slower than the binary endpoint and sends 4.9× more bytes: parsing
  128 numbers per vector dominates (the Flat insert itself is cheap). Clients that load large
  datasets should use `vectors:bulk`; the JSON endpoint is for convenience and small batches.
- Single run pairs on a laptop without thermal control; differences of ~10% between the two runs
  are noise.

## Reproduce

```powershell
. .\tools\dev-env.ps1
cmake --preset msvc-release; cmake --build --preset msvc-release
out\build\msvc-release\benchmarks\vf_http_bench.exe --n 100000 --dim 128 --queries 2000 --ingest 100000 --batch 1000 --out http_hnsw_100k_d128.json
```
