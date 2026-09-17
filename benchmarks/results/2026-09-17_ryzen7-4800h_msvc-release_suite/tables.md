### Exact search (queries/s, k = 10)

| N | tier | threads | d=128 | d=384 | d=768 | d=1536 |
|---|---|---|---|---|---|---|
| 10 000 | scalar | 1 | 1 290.0 | 368.5 | 182.0 | 90.0 |
| 10 000 | scalar | 16 | 11 849.5 | 3 138.7 | 1 436.9 | 640.5 |
| 10 000 | avx2 | 1 | 4 135.8 | 1 374.2 | 663.3 | 315.8 |
| 10 000 | avx2 | 16 | 29 179.9 | 6 285.4 | 1 987.6 | 744.0 |
| 100 000 | scalar | 1 | 128.9 | 37.3 | 18.1 | 9.1 |
| 100 000 | scalar | 16 | 825.4 | 262.0 | 118.4 | 59.9 |
| 100 000 | avx2 | 1 | 306.2 | 129.4 | 65.6 | 34.0 |
| 100 000 | avx2 | 16 | 902.2 | 279.9 | 135.7 | 64.2 |
| 1 000 000 | scalar | 1 | 13.2 | 3.7 | 1.8 | — |
| 1 000 000 | scalar | 16 | 40.5 | 11.5 | 5.5 | — |
| 1 000 000 | avx2 | 1 | 38.4 | 12.6 | 6.8 | — |
| 1 000 000 | avx2 | 16 | 96.8 | 31.3 | 12.4 | — |

### HNSW parameter sweep (100K × 128-d, one thread)

| M | efC | k | build s (1 thread) | graph MiB | QPS @ recall≥0.90 | QPS @ ≥0.95 | QPS @ ≥0.99 |
|---|---|---|---|---|---|---|---|
| 8 | 100 | 10 | 9.0 | 10 | 14 112 (ef 64) | 8 773 (ef 128) | 3 834 (ef 512) |
| 8 | 200 | 10 | 13.7 | 10 | 13 745 (ef 64) | 8 502 (ef 128) | 5 521 (ef 256) |
| 8 | 400 | 10 | 22.9 | 10 | 14 208 (ef 64) | 8 840 (ef 128) | 5 778 (ef 256) |
| 16 | 100 | 10 | 13.1 | 18 | 15 737 (ef 32) | 10 213 (ef 64) | 7 130 (ef 128) |
| 16 | 200 | 1 | 17.6 | 18 | 15 914 (ef 32) | 15 914 (ef 32) | 10 332 (ef 64) |
| 16 | 200 | 10 | 18.2 | 18 | 15 642 (ef 32) | 10 349 (ef 64) | 10 349 (ef 64) |
| 16 | 200 | 100 | 17.9 | 18 | 7 873 (ef 10) | 7 873 (ef 10) | 7 873 (ef 10) |
| 16 | 400 | 10 | 27.9 | 18 | 15 832 (ef 32) | 10 363 (ef 64) | 10 363 (ef 64) |
| 32 | 100 | 10 | 27.3 | 34 | 16 689 (ef 16) | 11 398 (ef 32) | 11 398 (ef 32) |
| 32 | 200 | 10 | 32.2 | 34 | 16 438 (ef 16) | 11 399 (ef 32) | 11 399 (ef 32) |
| 32 | 400 | 10 | 43.3 | 34 | 16 500 (ef 16) | 11 436 (ef 32) | 11 436 (ef 32) |
| 48 | 100 | 10 | 45.6 | 50 | 16 994 (ef 10) | 13 060 (ef 16) | 10 000 (ef 32) |
| 48 | 200 | 10 | 50.7 | 50 | 16 883 (ef 10) | 13 467 (ef 16) | 9 867 (ef 32) |
| 48 | 400 | 10 | 63.9 | 50 | 17 004 (ef 10) | 13 507 (ef 16) | 9 801 (ef 32) |

### HNSW scale (M = 16, efC = 200, 16 build threads)

| N | d | build s (16 threads) | graph MiB | peak RSS GiB | recall@10 ef=64 | ef for ≥0.95 | unreachable |
|---|---|---|---|---|---|---|---|
| 10 000 | 128 | 0.1 | 9 | 0.03 | 1.0000 | 10 | 0 |
| 10 000 | 384 | 0.3 | 9 | 0.05 | 1.0000 | 10 | 0 |
| 10 000 | 768 | 0.8 | 10 | 0.08 | 1.0000 | 10 | 0 |
| 10 000 | 1536 | 1.8 | 10 | 0.13 | 1.0000 | 10 | 0 |
| 100 000 | 128 | 2.5 | 22 | 0.13 | 0.9902 | 64 | 0 |
| 100 000 | 384 | 5.9 | 22 | 0.32 | 0.9762 | 64 | 1 |
| 100 000 | 768 | 12.5 | 22 | 0.61 | 0.9698 | 64 | 1 |
| 100 000 | 1536 | 23.4 | 22 | 1.18 | 0.9680 | 64 | 4 |
| 1 000 000 | 128 | 96.8 | 173 | 1.21 | 0.7908 | 256 | 108 |
| 1 000 000 | 384 | 182.2 | 173 | 3.12 | 0.7098 | 512 | 583 |
| 1 000 000 | 768 | 402.6 | 175 | 5.98 | 0.6732 | 512 | 1063 |

### Thread scaling (speedup vs one thread)

| workload | 1 T | 2 T | 4 T | 8 T | 12 T | 16 T |
|---|---|---|---|---|---|---|
| build-hnsw-100k-d128 | 1.00× | 1.96× | 3.68× | 5.71× | — | 7.36× |
| search-flat-100k-d128 | 1.00× | 1.88× | 3.02× | 4.71× | 6.13× | 2.70× |
| search-hnsw-100k-d128 | 1.00× | 1.91× | 3.14× | 4.15× | 4.43× | 4.71× |

### Search latency during ingestion

| mode / writer threads | vectors/s | idle p50 ms | busy p50 ms | busy p99.9 ms | busy max ms |
|---|---|---|---|---|---|
| coarse-w1 | 3 738 | 0.067 | 2.278 | 2.89 | 3.32 |
| concurrent-w1 | 4 388 | 0.072 | 0.087 | 0.30 | 2.63 |
| concurrent-w4 | 15 508 | 0.067 | 0.098 | 0.42 | 2.51 |

### Loading (warm file cache)

| run | open ms | first query ms | steady QPS | RSS after open MiB |
|---|---|---|---|---|
| load-flat-1m-d768-heap | 3709 | 147.99 | 7 | +3001 |
| load-flat-1m-d768-mmap | 168 | 831.99 | 6 | +81 |
| load-flat-1m-d768-mmap-prefault | 368 | 805.65 | 6 | +81 |
| load-hnsw-100k-d768-heap | 409 | 0.17 | 10 719 | +319 |
| load-hnsw-100k-d768-mmap | 61 | 0.77 | 10 802 | +50 |

### Saving

| run | build s | build threads | save s | file MiB |
|---|---|---|---|---|
| save-flat-100k-d128 | 0.0 | 1 | 0.08 | 50 |
| save-flat-1m-d768 | 1.2 | 1 | 4.95 | 2937 |
| save-hnsw-100k-d128 | 2.7 | 16 | 0.11 | 63 |
| save-hnsw-100k-d768 | 12.6 | 16 | 0.45 | 308 |

### Not run

| scenario | reason |
|---|---|
| exact-n1000000-d1536 | 6.1 GB of vectors plus the 6.1 GB generated input exceed what fits next to the OS in 15.4 GB RAM without paging. |
| scale-n1000000-d1536 | Same memory limit as exact-n1000000-d1536 (input, exact ground-truth copy and index). |
| datasets-real | SIFT1M, GloVe-100 and the 384-d/1536-d embedding sets are downloads that DESIGN §16.2 requires the user to approve; they were not downloaded. tools/datasets/fetch.py and hdf5_to_npy.py prepare them. |
| storage-cold-cache | Windows offers no supported way to evict the file cache without a reboot; all load timings are warm-cache. |
| threads-pinning | Thread affinity is not implemented in vf_bench; the OS scheduler places threads. |
| hnsw-sweep-n1000000 | The 12-configuration grid runs serial builds (graph statistics come from the single-threaded backend); at 1M rows this would take several hours. The 1M point is covered by scale-n1000000-* with a parallel build. |
| external-reference | hnswlib/FAISS comparisons are optional (DESIGN §16.1) and were not installed. |
