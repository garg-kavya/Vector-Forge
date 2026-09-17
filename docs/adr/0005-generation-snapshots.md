# ADR-0005: Generation files and a MANIFEST for durable snapshots

- **Status:** Accepted
- **Date:** 2026-09-14 (recorded 2026-09-17)
- **Related:** docs/DESIGN.md §12.4, docs/storage-format.md, `src/collection/snapshot.hpp`,
  `src/collection/catalog.cpp`

## Context

Collections are loaded with memory-mapped vectors. On Windows a mapped file cannot be replaced or
deleted, so "write a temporary file and rename it over the index" fails while the index is in use.
A crash at any point must leave a loadable collection. Collections created but never saved must
survive restarts too.

## Decision

- Each collection directory holds numbered generation files `index.NNNNNN.vfidx` and a small
  `MANIFEST` naming the current one with its header checksum. A snapshot writes generation N+1
  (atomic temp + sync + rename), then atomically replaces `MANIFEST`, then deletes older
  generations where possible; files still mapped are removed by a later snapshot or at startup.
- `config.json` records the configuration so a never-snapshotted collection is recreated empty.
- Creation writes into `.<name>.creating` and renames; a drop writes a `DROPPED` marker first;
  startup removes both kinds of leftovers.

## Alternatives considered

| Alternative | Why not chosen |
|---|---|
| Replace one index file in place | Impossible on Windows while mapped; a crash mid-write loses the index. |
| Write-ahead log of inserts | Durable per insert, but a large feature (log format, replay, compaction) outside v1; snapshots are explicit and documented as the durability point. |
| Copy-on-load (never map) | Loses the memory and load-time benefits measured in Phase 4. |

## Consequences

- Inserts are durable only after a snapshot (`POST …/snapshot`, `--snapshot-on-exit`); this is
  documented in the HTTP API page.
- Disk usage temporarily doubles for the collection during a snapshot while an old generation is
  still mapped.

## Validation

`tests/persistence/test_atomic_save.cpp` (fault injection at every step),
`tests/integration/test_catalog.cpp` (restart recovery, durable drops, damaged files),
`tests/http/test_routes.cpp` (restart from a snapshot).
