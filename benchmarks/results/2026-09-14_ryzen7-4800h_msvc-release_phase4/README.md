# Phase 4 storage baselines — 2026-09-14

Raw data (vf_bench JSON, one fresh process per file):

| File | Content |
|---|---|
| [`save_hnsw_100k_d128.json`](save_hnsw_100k_d128.json) | build + save, HNSW 100K × 128-d |
| [`load_hnsw_100k_d128_{heap,mmap,mmap_prefault}.json`](load_hnsw_100k_d128_heap.json) | load + two query passes |
| [`save_flat_1m_d768.json`](save_flat_1m_d768.json) | build + save, Flat 1M × 768-d (2.87 GiB file) |
| [`load_flat_1m_d768_{heap,mmap,mmap_prefault}.json`](load_flat_1m_d768_heap.json) | load + two query passes |

## Environment

| Item | Value |
|---|---|
| CPU | AMD Ryzen 7 4800H (Zen 2), 8 cores / 16 threads |
| RAM / disk | 15.4 GB; NVMe SSD, NTFS |
| OS | Windows 11 Home 10.0.26200 |
| Power | AC power, battery 100%, custom power plan |
| Compiler | MSVC 19.50.35728, `msvc-release` preset |
| SIMD tier | scalar; single thread |
| Source | `vf_git_sha` = `d2a88818d343-dirty`: the working tree that became the Phase 4 commit |

**Protocol and limits.** One run per configuration. Each load ran in a new process right after the
file was written, so the OS page cache was **warm**; Windows offers no practical way to evict it
(DESIGN §16.4), so cold-cache numbers were not measured. "First query" and pass 1 include the page
faults that map file pages into the process. RSS = working set; private = commit charge
(`PrivateUsage`). HNSW queries: 1 000, k = 10, ef_search = 64. Flat queries: 20, k = 10 (each scans
1M × 768 floats). Data: seeded Gaussian mixture (`vf_bench` defaults).

## Reproduce

```powershell
$b = "out\build\msvc-release\benchmarks\vf_bench.exe"
& $b --scenario save --n 100000 --dim 128 --metric l2 --index hnsw --index-file hnsw.vfidx --out save_hnsw_100k_d128.json
& $b --scenario load --index-file hnsw.vfidx --load-mode heap --queries 1000 --ef-search 64 --out load_hnsw_100k_d128_heap.json
& $b --scenario load --index-file hnsw.vfidx --load-mode mmap --queries 1000 --ef-search 64 --out load_hnsw_100k_d128_mmap.json
& $b --scenario load --index-file hnsw.vfidx --load-mode mmap --prefault --queries 1000 --ef-search 64 --out load_hnsw_100k_d128_mmap_prefault.json
& $b --scenario save --n 1000000 --dim 768 --metric l2 --index flat --index-file flat.vfidx --out save_flat_1m_d768.json
& $b --scenario load --index-file flat.vfidx --load-mode heap --queries 20 --out load_flat_1m_d768_heap.json
& $b --scenario load --index-file flat.vfidx --load-mode mmap --queries 20 --out load_flat_1m_d768_mmap.json
& $b --scenario load --index-file flat.vfidx --load-mode mmap --prefault --queries 20 --out load_flat_1m_d768_mmap_prefault.json
```

## Save

| Index | File | Build | Save | Throughput |
|---|---|---|---|---|
| HNSW 100K × 128 | 63.5 MiB | 41.0 s | 0.111 s | 573 MiB/s |
| Flat 1M × 768 | 2 937 MiB | 1.9 s | 8.44 s | 348 MiB/s |

Save includes CRC computation, `FlushFileBuffers` and the atomic rename.

## Load — Flat 1M × 768 (2.87 GiB)

| Mode | Open | First query | Pass 1 p99 | Pass 2 QPS | Private after open | RSS after queries |
|---|---|---|---|---|---|---|
| heap (full verify) | 5.64 s | 636 ms | 638 ms | 1.8 | 3 021 MiB | 3 007 MiB |
| mmap (metadata verify) | 0.198 s | 1 253 ms | 1 253 ms | 1.8 | 92 MiB | 3 015 MiB |
| mmap + prefault | 0.377 s | 1 241 ms | 1 241 ms | 1.8 | 92 MiB | 3 015 MiB |

## Load — HNSW 100K × 128

| Mode | Open | First query | Pass 1 p50 / p99 | Pass 2 QPS | Pass 2 p50 / p99 | Private after open |
|---|---|---|---|---|---|---|
| heap (full verify) | 0.119 s | 288 µs | 225 / 297 µs | 5 364 | 184 / 254 µs | 90 MiB |
| mmap (metadata verify) | 0.056 s | 1 184 µs | 190 / 625 µs | 5 155 | 188 / 274 µs | 42 MiB |
| mmap + prefault | 0.061 s | 1 066 µs | 188 / 595 µs | 5 269 | 186 / 251 µs | 42 MiB |

(48 MiB of the HNSW vectors were mapped: three full 16 MiB chunks; the remaining 1 696 rows were
copied.)

## Observations (this machine, warm page cache)

1. **mmap makes opening large indexes cheap:** 0.20 s instead of 5.64 s for 2.87 GiB (28× faster),
   and the process commits 92 MiB instead of 3 021 MiB because vector pages belong to the file cache.
   Resident memory converges once every page has been touched (3 015 MiB after the scans).
2. **The cost moves to the first touches:** the first Flat query took 1.25 s with mmap versus 0.64 s on
   the heap copy, because it faults in the whole file; HNSW pass-1 p99 was 625 µs versus 297 µs.
   Steady-state throughput is the same in both modes (1.8 vs 1.8 QPS Flat; 5 155–5 364 QPS HNSW,
   within single-run variation).
3. **Prefault did not help here:** `PrefetchVirtualMemory` returned quickly (+0.18 s open) but the first
   Flat query was not faster (1.24 s). Whether it pays off with a cold cache or on Linux
   (`MADV_WILLNEED`) is untested.
4. **Heap loads verify everything:** the 5.64 s include CRCs over all 2.87 GiB, finiteness checks and
   the copy; the default mmap load (`Verify::Metadata`) does not read the vector section at all.
5. These are single warm-cache runs on a laptop. Cold-cache load latency and multi-process page-cache
   sharing remain unmeasured.
