# VectorForge index file format (`.vfidx`, format 1.0)

This page specifies the on-disk format written by `Collection::save` and read by `Collection::load`,
the validation the reader performs, and the snapshot directory layout. Design rationale is in
[DESIGN.md §12](DESIGN.md#12-storage-design).

An independent Python writer, [`tools/make_golden.py`](../tools/make_golden.py), implements this
specification; the files it produces live in `tests/data/golden/` and the C++ tests load them and
re-save them byte for byte.

## Conventions

- All integers are **little-endian**, fixed width. Floating-point values are IEEE 754 binary32 in
  little-endian byte order. Big-endian hosts are not supported (compile-time check).
- Records are encoded field by field; there is no struct padding other than the fields listed.
- Every byte of the file belongs to the header, a section, the section table, or zero padding.
- A file is **canonical** when it is laid out exactly as described under "Canonical layout".
  `save` always writes canonical files, so saving the same state twice gives identical bytes.

## File layout

```
offset 0        FileHeader (64 bytes)
offset 64..     sections, each at an offset that is a multiple of 8 (VECTORS: of 4096)
                zero padding between regions
table_offset    SectionTable: section_count x SectionEntry (32 bytes each)
file_size       end of file
```

### FileHeader (64 bytes)

| Offset | Type | Field | Value / meaning |
|---|---|---|---|
| 0 | 8 bytes | magic | `56 46 49 44 58 0D 0A 1A` = `"VFIDX\r\n\x1A"` (detects text-mode corruption) |
| 8 | u16 | format_major | 1. Readers refuse other majors (`UnsupportedVersion`). |
| 10 | u16 | format_minor | 0. Minor versions only add optional sections. |
| 12 | u32 | endian_tag | `0x01020304`; `0x04030201` means a big-endian file (`UnsupportedVersion`) |
| 16 | u64 | flags | bit 0 `HAS_GRAPH`, bit 1 `HAS_TOMBSTONES`, bit 2 `VECTORS_NORMALIZED`; other bits must be 0 |
| 24 | u64 | file_size | total size in bytes (detects truncation and extension) |
| 32 | u64 | section_table_offset | multiple of 8 |
| 40 | u32 | section_count | 1..64 |
| 44 | u32 | section_table_crc32c | CRC-32C of the section table bytes |
| 48 | u64 | reserved | 0 |
| 56 | u32 | header_crc32c | CRC-32C of header bytes 0..55 |
| 60 | u32 | reserved | 0 |

CRC-32C is the Castagnoli CRC (reflected polynomial `0x82F63B78`, initial value and final XOR
`0xFFFFFFFF`), check value `crc32c("123456789") = 0xE3069283`.

### SectionEntry (32 bytes)

| Offset | Type | Field |
|---|---|---|
| 0 | u32 | type |
| 4 | u32 | flags: bit 0 `OPTIONAL` (a reader that does not know the type may skip it); other bits 0 |
| 8 | u64 | offset (multiple of 8) |
| 16 | u64 | size in bytes |
| 24 | u32 | crc32c of the section bytes |
| 28 | u32 | reserved, 0 |

### Sections

With `N = METADATA.node_count` (rows, including deleted ones), `D = dim`, `M`, `M0 = 2M`:

| Type | Name | Presence | Size | Content |
|---|---|---|---|---|
| 1 | METADATA | required | variable | see below |
| 2 | VECTORS | required | `N * D * 4` | row-major float32 rows; offset is a multiple of 4096 so a whole-file mapping yields aligned rows |
| 3 | LABELS | required | `N * 8` | u64 external id of every row |
| 4 | TOMBSTONES | iff `HAS_TOMBSTONES` | `ceil(N / 64) * 8` | u64 bitset words; bit `r % 64` of word `r / 64` marks row `r` deleted; bits at or beyond `N` are 0; at least one bit set |
| 5 | LEVELS | iff `HAS_GRAPH` | `N` | u8 top level of every node |
| 6 | L0_LINKS | iff `HAS_GRAPH` | `N * (1 + M0) * 4` | per node: u32 count, then `M0` u32 slots; slots at or beyond count are `0xFFFFFFFF` |
| 7 | UPPER_INDEX | iff `HAS_GRAPH` | `N * 8` | u64 word offset of the node's block in UPPER_LINKS, or `0xFFFFFFFFFFFFFFFF` for level-0 nodes |
| 8 | UPPER_LINKS | iff `HAS_GRAPH` | `sum(level * (1 + M)) * 4` | per node with level `l >= 1`, in node order: `l` lists (levels 1..l), each u32 count and `M` slots (unused `0xFFFFFFFF`) |

`HAS_GRAPH` is set exactly for HNSW indexes; `VECTORS_NORMALIZED` exactly when the configuration
normalises (`normalize` or cosine metric), in which case the stored rows have unit length.

### METADATA

| Offset | Type | Field |
|---|---|---|
| 0 | u32 | dim (1..65536) |
| 4 | u8 | metric: 0 L2, 1 inner product, 2 cosine |
| 5 | u8 | index type: 0 Flat, 1 HNSW |
| 6 | u8 | normalize: 0 or 1 (as configured; cosine normalises regardless) |
| 7 | u8 | 0 |
| 8 | u64 | node_count (rows, < 2^32 - 1) |
| 16 | u64 | live_count (rows not deleted) |
| 24 | u32 | M |
| 28 | u32 | M0 (= 2M) |
| 32 | u32 | ef_construction |
| 36 | u32 | ef_search (default query beam) |
| 40 | u8 | max_level_cap (`HnswParams::max_level`, <= 32) |
| 41 | u8 | max_level: level of the entry point (0 for Flat) |
| 42 | u16 | 0 |
| 44 | u32 | entry_point: node id, or `0xFFFFFFFF` for Flat and empty HNSW indexes |
| 48 | u64 | seed (level generation) |
| 56 | u64 | reserved, 0 (DESIGN.md's `mL`: levels are computed exactly from M, so it is not stored) |
| 64 | u16 | creator length (<= 256) |
| 66 | bytes | creator (e.g. `vectorforge 0.1.0`): the version that created the collection |
| 66 + len | u64 | created_unix_ms: creation time of the collection |

HNSW parameters are stored for Flat indexes too, so configurations round-trip. Creator and creation time are preserved by
load and save, which keeps re-saved files byte-identical.

## Canonical layout

Sections appear in the order METADATA, LABELS, TOMBSTONES (if any), LEVELS, L0_LINKS, UPPER_INDEX,
UPPER_LINKS (HNSW), VECTORS. Each starts at the next multiple of 8 after the previous region
(VECTORS: the next multiple of 4096), followed by the section table at the next multiple of 8, which
ends the file. Padding bytes are zero, unused link slots are `0xFFFFFFFF`, UPPER_INDEX offsets are
cumulative in node order, and LABELS of deleted rows keep the label they had.

## Validation

`Collection::load` treats the file as untrusted input. All arithmetic on offsets and sizes is
overflow-checked; no check is an assertion; failures return `CorruptData` naming the section and
field, or `UnsupportedVersion`. In order:

1. **Header:** size >= 64, magic, endian tag, major version, header CRC, reserved fields zero,
   `file_size` equals the actual size, no unknown flags, `section_count` in 1..64.
2. **Section table:** in bounds, CRC, entries well formed, every section in bounds at a multiple of
   8, no unknown flags, no duplicate types, unknown types allowed only if `OPTIONAL` (then skipped),
   regions do not overlap, all other bytes zero.
3. **Presence:** METADATA, VECTORS and LABELS present; TOMBSTONES iff `HAS_TOMBSTONES`; graph
   sections iff `HAS_GRAPH`; VECTORS offset page-aligned.
4. **Checksums** of all sections except VECTORS (unless `Verify::None`).
5. **METADATA:** field ranges, padding and reserved fields zero, flags consistent with metric and
   index type, HNSW parameters valid (`HnswParams::validate`, `M0 == 2M`), Flat has no entry point.
6. **Sizes:** every section exactly as derived from METADATA.
7. **VECTORS checksum** (`Verify::Full`).
8. **TOMBSTONES and LABELS:** no bits beyond the last row, `live_count` consistent, live labels
   unique and not `0xFFFFFFFFFFFFFFFF`.
9. **Graph** (linear pass, O(N * M0)): levels <= cap, entry point on the top level, canonical
   UPPER_INDEX, every list with count <= capacity, ids < N, no self-links, no duplicates, linked
   nodes at least at the list's level, unused slots `0xFFFFFFFF`.
10. **Vector values:** finite and within `kMaxAbsComponent`, whenever vectors are copied, and for
    mapped vectors with `Verify::Full`.

| `LoadOptions::verify` | Checksums verified |
|---|---|
| `Auto` (default) | `Full` for heap loads, `Metadata` for mmap loads |
| `Full` | everything |
| `Metadata` | everything except VECTORS (keeps mmap loads lazy) |
| `None` | header and section table only |

Structural validation (all steps except 4 and 7) always runs, so a damaged file cannot cause an
out-of-bounds read or an invalid graph traversal whatever the verification level. With `Metadata` or
`None`, damage inside VECTORS may go unnoticed and produce wrong distances; it cannot crash.

All allocations while loading are bounded by a small multiple of the file size.

## Loading modes

- **mmap** (`use_mmap = true`, default): the whole file is mapped read-only; vector chunks of the
  store point into the mapping (whole 16 MiB chunks only; the last partial chunk is copied so later
  inserts continue in heap chunks). Graph, labels and tombstones are copied to the heap. The file
  must not be modified or truncated while the collection lives. On Windows it also cannot be
  replaced (renaming over it fails); deleting it works on Windows 10 and later.
- **heap**: the file is still parsed through a temporary mapping (page cache, not a private copy),
  and all data is copied; the mapping is released before `load` returns.

## Atomic save

`Collection::save(path)` writes `path.tmp`, syncs it (`fsync` / `FlushFileBuffers`), renames it over
`path` (`rename` / `MoveFileExW(MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)`) and, on POSIX,
syncs the directory. A crash leaves either the old or the new file. Tests inject a simulated crash
after each of these four steps.

## Snapshots (generations)

For long-running services (the Phase 7 catalog), a collection is saved as numbered generations in a
directory:

```
<dir>/MANIFEST              {"format": 1, "generation": 42, "file": "index.000042.vfidx", "crc32c": 1234567890}
<dir>/index.000042.vfidx
```

- `crc32c` is the `header_crc32c` of the generation file; it transitively covers the whole file.
- A snapshot writes `index.<N+1>.vfidx` atomically, then replaces `MANIFEST` atomically, then deletes
  other generations and leftover `.tmp` files where possible. The manifest rename is the commit point.
- The MANIFEST parser is strict: exactly these four keys, unsigned integers, no escapes, and `file`
  must equal the canonical name for `generation` (so it can never name another path).
- Garbage collection only removes `index.<digits>.vfidx[.tmp]` other than the current generation and
  `MANIFEST.tmp`; files it cannot delete are counted and retried later.

## Compatibility

- Readers of major 1 accept any minor version and skip unknown optional sections.
- New fields are added only as new optional sections; METADATA is not extended within major 1.
- Golden files for every released `(major, minor)` are kept in `tests/data/golden/` and must keep
  loading.

## Tooling

```bash
vectorforge build  --input base.fvecs --metric l2 --index hnsw --M 16 --ef-construction 200 --out idx.vfidx
vectorforge info   idx.vfidx     # header, sections, parameters, level histogram
vectorforge verify idx.vfidx     # full checksums + graph invariants; non-zero exit on failure
vectorforge search --index idx.vfidx --queries query.fvecs --k 10 --ef 100 --gt groundtruth.ivecs
```

Save and heap/mmap load measurements: [benchmarks/results/2026-09-14_ryzen7-4800h_msvc-release_phase4](../benchmarks/results/2026-09-14_ryzen7-4800h_msvc-release_phase4/README.md).

The loader is fuzzed with libFuzzer (`tests/fuzz/fuzz_index_reader.cpp`, preset `linux-clang-fuzz`).
