#!/usr/bin/env python3
"""Generate the golden .vfidx files in tests/data/golden/.

This is an independent implementation of the index file format (docs/storage-format.md) in pure
Python (standard library only). The C++ test suite loads these files, checks their contents and
search results, and re-saves them byte for byte, so the writer, the reader and the specification
are cross-checked against each other.

Usage:  python tools/make_golden.py [output_dir]

The generated files are committed. Regenerate only when adding files for a new format version;
never change existing golden files, since they pin backward compatibility.
"""

from __future__ import annotations

import math
import pathlib
import struct
import sys

MAGIC = b"VFIDX\r\n\x1a"
ENDIAN_TAG = 0x01020304
FLAG_HAS_GRAPH = 1 << 0
FLAG_HAS_TOMBSTONES = 1 << 1
FLAG_VECTORS_NORMALIZED = 1 << 2
EMPTY_SLOT = 0xFFFFFFFF
NO_UPPER_BLOCK = 0xFFFFFFFFFFFFFFFF

METADATA, VECTORS, LABELS, TOMBSTONES, LEVELS, L0_LINKS, UPPER_INDEX, UPPER_LINKS = range(1, 9)
METRIC = {"l2": 0, "ip": 1, "cosine": 2}
INDEX = {"flat": 0, "hnsw": 1}

CREATOR = b"vectorforge-golden 1.0"
CREATED_UNIX_MS = 1_700_000_000_000


def crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0x82F63B78 if crc & 1 else crc >> 1
    return crc ^ 0xFFFFFFFF


def align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def integer_vector(i: int, dim: int) -> list[float]:
    """Deterministic small-integer vectors (exact in float32)."""
    return [float(((i * 7 + j * 3 + i * j) % 11) - 5) for j in range(dim)]


def normalized(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def write_index(path: pathlib.Path, *, dim, metric, index, normalize, vectors, labels,
                deleted, m=16, ef_construction=200, ef_search=50, max_level_cap=16, seed=0,
                levels=None, l0=None, upper=None, entry_point=EMPTY_SLOT, max_level=0):
    n = len(vectors)
    vectors_normalized = normalize or metric == "cosine"
    words = (n + 63) // 64
    tombstones = [0] * words
    for r in deleted:
        tombstones[r // 64] |= 1 << (r % 64)
    live_count = n - len(deleted)

    metadata = struct.pack("<IBBBBQQIIIIBBHIQQ", dim, METRIC[metric], INDEX[index],
                           1 if normalize else 0, 0, n, live_count, m, 2 * m, ef_construction,
                           ef_search, max_level_cap, max_level, 0, entry_point, seed, 0)
    metadata += struct.pack("<H", len(CREATOR)) + CREATOR + struct.pack("<Q", CREATED_UNIX_MS)

    sections = [(METADATA, metadata, 8), (LABELS, struct.pack(f"<{n}Q", *labels), 8)]
    if deleted:
        sections.append((TOMBSTONES, struct.pack(f"<{words}Q", *tombstones), 8))
    if index == "hnsw":
        def list_words(ids, capacity):
            return [len(ids)] + list(ids) + [EMPTY_SLOT] * (capacity - len(ids))

        l0_words = []
        for i in range(n):
            l0_words += list_words(l0[i], 2 * m)
        upper_index = []
        upper_words = []
        for i in range(n):
            if levels[i] == 0:
                upper_index.append(NO_UPPER_BLOCK)
                continue
            upper_index.append(len(upper_words))
            for level in range(1, levels[i] + 1):
                upper_words += list_words(upper[(i, level)], m)
        sections += [
            (LEVELS, bytes(levels), 8),
            (L0_LINKS, struct.pack(f"<{len(l0_words)}I", *l0_words), 8),
            (UPPER_INDEX, struct.pack(f"<{n}Q", *upper_index), 8),
            (UPPER_LINKS, struct.pack(f"<{len(upper_words)}I", *upper_words), 8),
        ]
    flat = [x for v in vectors for x in v]
    sections.append((VECTORS, struct.pack(f"<{len(flat)}f", *flat), 4096))

    body = bytearray(64)
    table = bytearray()
    for section_type, payload, alignment in sections:
        body += bytes(align(len(body), alignment) - len(body))
        table += struct.pack("<IIQQII", section_type, 0, len(body), len(payload), crc32c(payload), 0)
        body += payload
    body += bytes(align(len(body), 8) - len(body))
    table_offset = len(body)
    body += table

    flags = ((FLAG_HAS_GRAPH if index == "hnsw" else 0) | (FLAG_HAS_TOMBSTONES if deleted else 0) |
             (FLAG_VECTORS_NORMALIZED if vectors_normalized else 0))
    header = MAGIC + struct.pack("<HHIQQQII", 1, 0, ENDIAN_TAG, flags, len(body), table_offset,
                                 len(sections), crc32c(bytes(table))) + bytes(8)
    header += struct.pack("<II", crc32c(header), 0)
    assert len(header) == 64
    body[0:64] = header
    path.write_bytes(bytes(body))


def main() -> None:
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "tests/data/golden")
    out.mkdir(parents=True, exist_ok=True)

    # Flat, L2, 12 rows of dimension 4, labels 100 + 3i, no deletions.
    write_index(out / "v1.0_flat_l2.vfidx", dim=4, metric="l2", index="flat", normalize=False,
                vectors=[integer_vector(i, 4) for i in range(12)],
                labels=[100 + 3 * i for i in range(12)], deleted=[])

    # HNSW, cosine, 16 rows of dimension 3, M = 2, rows 3 and 10 deleted. Hand-built graph: a ring on
    # level 0, a ring through the level-1 nodes on level 1, and node 0 alone on level 2.
    n = 16
    levels = [2, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
    l0 = [[(i + 1) % n, (i - 1) % n] for i in range(n)]
    ring1 = [i for i in range(n) if levels[i] >= 1]
    upper = {}
    for position, node in enumerate(ring1):
        upper[(node, 1)] = [ring1[(position + 1) % len(ring1)], ring1[(position - 1) % len(ring1)]]
    upper[(0, 2)] = []
    write_index(out / "v1.0_hnsw_cosine.vfidx", dim=3, metric="cosine", index="hnsw", normalize=False,
                vectors=[normalized(integer_vector(i + 1, 3)) for i in range(n)],
                labels=[5000 + i for i in range(n)], deleted=[3, 10], m=2, ef_construction=8,
                ef_search=16, max_level_cap=4, seed=12345, levels=levels, l0=l0, upper=upper,
                entry_point=0, max_level=2)

    # Empty HNSW collection, inner product, dimension 7.
    write_index(out / "v1.0_hnsw_ip_empty.vfidx", dim=7, metric="ip", index="hnsw", normalize=False,
                vectors=[], labels=[], deleted=[], m=4, ef_construction=16, ef_search=8,
                max_level_cap=8, seed=1, levels=[], l0=[], upper={})


if __name__ == "__main__":
    main()
