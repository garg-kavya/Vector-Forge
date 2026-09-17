# ADR-0001: Status/Result for expected errors, exceptions only for allocation failure

- **Status:** Accepted
- **Date:** 2026-09-13 (recorded 2026-09-17)
- **Related:** docs/DESIGN.md §4.7, `include/vectorforge/status.hpp`

## Context

The library is used from C++, an HTTP server and Python. Most failures are expected and must be
reported precisely: invalid vectors, unknown ids, damaged files, I/O errors. Hot paths (search,
insert) must not pay for error handling, and failures must map cleanly to HTTP status codes and
Python exceptions. Out-of-memory can happen anywhere a container grows.

## Decision

- Expected failures are returned as `vf::Status` (a code from a small stable enum plus a message)
  or `vf::Result<T>` (value or status, shaped like C++23 `std::expected`). An OK status holds no
  heap state. Every public function documents its error codes.
- `std::bad_alloc` propagates as an exception; the write paths give a documented guarantee
  (strong for single inserts, per-row for batches), checked by allocation-failure injection tests.
- Programming errors (violated preconditions) abort through `VF_CHECK`; `VF_ASSERT` checks are
  compiled into debug builds.
- Frontends map codes: HTTP status plus a stable string code (`to_string(ErrorCode)`), Python
  built-in or package exception classes.

## Alternatives considered

| Alternative | Why not chosen |
|---|---|
| Exceptions for all errors | Costly and non-local for expected conditions such as "id not found"; harder to map to HTTP codes exhaustively; noisy under `-fno-exceptions` consumers. |
| Error codes via output parameters | Easy to ignore; `[[nodiscard]]` on `Status`/`Result` makes ignoring an error a warning. |
| `std::expected` | C++23; the project targets C++20. `Result` mirrors its observers so a migration is mechanical. |
| Handling `bad_alloc` as a status | Every container growth would need a try/catch; the strong/per-row guarantees give callers the same recoverability. |

## Consequences

- Error paths are explicit in signatures and tested per code; frontends have one mapping table
  each.
- Accessing the value of an error `Result` aborts, so misuse fails loudly in tests.
- Code that grows containers after modifying state must be ordered carefully (grow first, then
  commit), which the insert paths do.

## Validation

`tests/unit/test_status.cpp`, `tests/alloc/test_exception_safety.cpp`,
`tests/http/test_routes.cpp` (error mapping), `python/tests/test_errors.py`.
