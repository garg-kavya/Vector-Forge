# ADR-0004: cpp-httplib and nlohmann/json for the HTTP API

- **Status:** Accepted
- **Date:** 2026-09-17
- **Related:** docs/DESIGN.md §13, docs/http-api.md

## Context

The server is a single-node front end whose request cost is dominated by search computation. It
must build on MSVC, MinGW, GCC and Clang, run under TSan/ASan, and stay out of the core library.
JSON is the expected interface; large ingestion needs a cheaper path.

## Decision

- cpp-httplib 0.54.1, used as a header-only target without TLS or compression (its own CMake
  project, which probes OpenSSL/zlib/brotli/zstd, is not configured). Blocking I/O on a fixed
  thread pool; the server adds its own compute pool for batch requests.
- nlohmann/json 3.12.0 for request decoding and response encoding, behind strict decoders
  (unknown keys rejected, a nesting-depth check before parsing, typed ids and numbers).
- A binary bulk-insert endpoint for high-volume ingestion.
- Both dependencies are fetched by tag and SHA-256; `VF_USE_SYSTEM_DEPS` switches to installed
  packages. Only `vf_server` links them.
- The server overrides cpp-httplib's default socket options: `SO_EXCLUSIVEADDRUSE` on Windows,
  `SO_REUSEADDR` without `SO_REUSEPORT` on POSIX, so a second process cannot bind the same port.

## Alternatives considered

| Alternative | Why not chosen |
|---|---|
| Boost.Beast / Asio | Asynchronous design and Boost dependency not justified by the workload. |
| Drogon, Crow, Oat++ | Framework lock-in and larger dependency trees. |
| gRPC | Heavy toolchain (protobuf, code generation); a possible later binary API. |
| simdjson / RapidJSON | Faster parsing, but request decoding is ~0.06 ms of a 0.33 ms loopback round trip for a 128-d query; the binary endpoint covers bulk data. |

## Consequences

- JSON ingestion is ~23× slower than the binary endpoint (measured), which the docs state.
- Header-only httplib increases the compile time of one translation unit and the tests.
- No TLS in-process: deployments put a TLS proxy in front (documented).

## Validation

`tests/http/*`, `tests/fuzz/fuzz_json_request.cpp`, `tools/check_openapi.py`,
`benchmarks/results/2026-09-17_ryzen7-4800h_msvc-release_phase7`.
