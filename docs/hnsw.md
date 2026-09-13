# HNSW in VectorForge

VectorForge's approximate index is a Hierarchical Navigable Small World graph, after Malkov & Yashunin,
*Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*
(IEEE TPAMI 2018, arXiv:1603.09320). This page describes the implementation as built in Phase 3
(single-threaded). The design rationale is in [DESIGN.md §9](DESIGN.md#9-hnsw-design).

Source: `src/index/hnsw/`, `src/search/{visited_set,search_context,context_pool}.hpp`,
`src/index/query_distance.hpp`.

## Using it

```cpp
vf::CollectionConfig cfg;
cfg.dim = 128;
cfg.metric = vf::Metric::Cosine;
cfg.index = vf::IndexType::Hnsw;   // the default
cfg.hnsw.M = 16;                   // links per node (2M on level 0)
cfg.hnsw.ef_construction = 200;    // build beam width, >= M
cfg.hnsw.ef_search = 50;           // default query beam width
auto col = vf::Collection::create(cfg).value();
// ... add / add_batch ...
vf::SearchParams p;
p.k = 10;
p.ef_search = 128;                 // per-query override; the beam is max(ef_search, k)
auto hits = col->search(query, p);
```

| Parameter | Effect |
|---|---|
| `M` | Out-degree target. Larger: better recall at a given `ef_search`, more memory (≈ 4·(1+2M) bytes/node on level 0) and slower builds. |
| `ef_construction` | Candidate beam while inserting. Larger: better graph, roughly linear build-time increase. |
| `ef_search` | Candidate beam while querying; the main recall/latency knob. |
| `max_level` | Cap on node levels (default 16). |
| `seed` | Level assignment seed; same data + parameters + seed ⇒ identical graph. |

## Storage

- **Level 0** — every node has a fixed-stride list `{count, ids[2M]}` in 64-byte-aligned chunks
  (about 16 MiB each); node `i`'s list is at `chunk[i >> shift] + (i & mask)·stride`.
- **Levels ≥ 1** — a node of level `l ≥ 1` owns a block of `l` lists `{count, ids[M]}` in an
  append-only arena; `upper[i]` holds the block offset.
- Addresses never change after allocation. Lists are read through `LinkView`, which is the only place
  that will change when Phase 6b makes link slots atomic.
- `HnswGraph::canonical_bytes()` encodes the logical graph (levels, entry point, the first `count` ids
  of every list). Tests use it to compare graphs until the Phase 4 file format exists.

## Levels

`level(id)` depends only on `(seed, id)`: `bits = splitmix64(seed ^ splitmix64(id))`,
`v = (bits >> 11) + 1 ∈ [1, 2^53]`, and the level is the largest `l ≤ max_level` with `M^l ≤ ⌊2^53 / v⌋`.
This is exactly the paper's `⌊−ln(U)·mL⌋` for `U = v·2^−53`, evaluated with integer arithmetic, so it is
identical on every compiler and C library and independent of insertion order.

## Search

1. Start at the entry point (a node of the highest level).
2. On each level above 0, move greedily to any neighbour that is closer, until none is.
3. On level 0, run a beam search with `ef = max(ef_search, k)`: a min-heap of candidates to expand, a
   bounded max-heap `W` of the best `ef` results, and an epoch-stamped visited array (2 bytes per node,
   O(1) reset). Stop when the nearest unexpanded candidate is worse than the worst result in a full `W`.
4. Return the first `k` of `W` in ascending `(distance, id)` order.

Every comparison uses the pair `(distance, internal id)`, so ties are resolved identically everywhere.
Removed (tombstoned) nodes are expanded like any other node but never enter `W`. With many removals `W`
fills more slowly, so queries do more work; compaction arrives in Phase 6a.

Queries lease a `SearchContext` from a mutex-protected pool; after the first queries have sized the
pooled contexts, a query performs no heap allocation.

## Insertion

Insertion follows the paper's Algorithm 1 in an explicit publication order (DESIGN §9.6):

1. **Compute (pure reads):** greedy descent to the node's level, then for every shared level a beam
   search with `ef_construction` over all nodes (tombstoned ones included) and neighbour selection
   (heuristic, Algorithm 4, keeping pruned candidates to refill up to `M`). All allocation happens here,
   so a failure leaves the graph unchanged.
2. **Own lists:** write the new node's lists; it is not reachable yet.
3. **Back-links:** append the node to each chosen neighbour's list, or re-select that list with the same
   heuristic when it is full (`2M` on level 0, `M` above).
4. **Entry point:** a node above the current top level becomes the new entry point.

Because distances are symmetric bit for bit (`dist(a, b) == dist(b, a)` for every metric formula), the
distance computed in step 1 is reused in step 3.

**Orphan repair.** If every chosen neighbour prunes the new node in step 3, nothing links to it on that
level. The node is then appended to the nearest construction candidate that still has a free slot. This
matters for duplicates: with 1 000 identical vectors, 966 nodes were unreachable without the repair
(measured by running `HnswDuplicates.IdenticalVectors` with `repair_orphans = false`) and 0 with it. On L2 and cosine clustered data it never triggered in our measurements, so those graphs are
exactly the paper's.

## Distances

| Metric | Stored rows | Distance |
|---|---|---|
| L2 | raw | `Σ(q−x)²` |
| InnerProduct | raw | `−⟨q, x⟩` |
| Cosine (or `normalize = true`) | unit length | cosine `max(0, 1 − ⟨q, x⟩/‖q‖)`, L2 `max(0, 2 − 2⟨q, x⟩/‖q‖)`, IP `−⟨q, x⟩/‖q‖` |

Queries are never copied; normalised collections scale the dot product by `1/‖q‖`. The formulas are shared
with the Flat backend, so both return bit-identical distances.

## Correctness checks

`HnswValidator` checks, for every list: `count ≤ capacity`, ids in range, no self-links, no duplicates,
every linked node has a level ≥ the list's level; and that the entry point exists iff the graph is
non-empty and has the top level. `reachability()` runs a BFS per level from the entry point, and
`level_histogram()` counts nodes per top level.

| Test | What it checks |
|---|---|
| `test_level_generator` | determinism, paper formula, golden values, cap, `P(level ≥ l) = M^−l` over 10⁶ ids |
| `test_visited_set` | visit/reset, growth, epoch wrap-around at 65 535; context pool reuse |
| `test_neighbor_select` | hand-built 2-D cases for heuristic, `keep_pruned`, simple selection, duplicates |
| `test_hnsw_graph` | storage across chunk boundaries, stable addresses, canonical encoding |
| `test_hnsw_validator` | each invariant violation is detected; per-level unreachable counts |
| `test_hnsw_edge_cases` | empty index, 1 element, k > count, ef < k, dim 1, delete entry point, delete all, reinsert, upsert, batch = single, identical and near-identical vectors |
| `test_hnsw_tombstones` | removal leaves the graph unchanged and deleted ids never appear; model-based test of 10⁴ random operations per metric (membership, distances, order, recall floor) |
| `test_hnsw_determinism` | same seed ⇒ same graph and results (also with different chunking); golden fingerprints |
| `test_hnsw_recall` (integration) | N = 10 000 clustered, d ∈ {16, 128}, L2/IP/cosine: validator, reachability, recall@10 monotone in `ef`, frozen thresholds |
| `test_zero_alloc_search` | HNSW `search_into` allocates nothing after warm-up |

### Test thresholds

The recall test data is generated without `std::log`, so it is bit-identical on every platform. Measured
mean recall@10 (200 queries, `M = 16`, `ef_construction = 200`; identical on MSVC 19.50, GCC 13.3,
GCC 15.2 and Clang 18) and the frozen thresholds:

| Case | ef = 32 measured | threshold | ef = 128 measured | threshold | unreachable (level 0) | bound |
|---|---|---|---|---|---|---|
| L2, d = 16 | 1.0000 | 0.98 | 1.0000 | 0.99 | 0 | 0 |
| IP, d = 16 | 0.9990 | 0.97 | 1.0000 | 0.99 | 503 | 603 |
| cosine, d = 16 | 1.0000 | 0.98 | 1.0000 | 0.99 | 0 | 0 |
| L2, d = 128 | 0.9995 | 0.97 | 1.0000 | 0.99 | 0 | 0 |
| IP, d = 128 | 1.0000 | 0.98 | 1.0000 | 0.99 | 223 | 267 |
| cosine, d = 128 | 0.9960 | 0.97 | 1.0000 | 0.99 | 0 | 0 |

Rule: threshold = measured − 0.02 at `ef = 32` and − 0.01 at `ef = 128`, rounded down to 0.01;
unreachable bound = measured + 20 % (IP only). Model-based test: measured recall 1.0000 (L2, cosine) and
0.9998 (IP), threshold 0.97. Tombstone test (30 % deleted): measured 1.0000, threshold 0.95. These are
regression guards, not performance claims.

## Known limitations

- **Inner product is not a metric.** With raw (unnormalised) vectors, large-norm vectors dominate
  every neighbourhood and pruning evicts low-norm nodes from all lists, so some nodes become unreachable
  and can never be returned: 503 of 10K (d = 16) and 223 of 10K (d = 128) clustered test nodes, and 38 850
  of 100K Gaussian-mixture nodes in `vf_bench`. Recall for in-distribution queries stayed high (≥ 0.995)
  because the true inner-product neighbours are mostly the large-norm vectors. If every vector must be
  findable, normalise (use cosine or `normalize = true`). A prototype repair based on in-degree counters
  reduced but did not remove the effect (118 → 6 at 10K, 38 850 → 21 053 at 100K) and was not adopted.
- The graph is never shrunk: removed vectors keep their nodes until compaction (Phase 6a).
- Single-threaded construction; concurrent insertion is Phase 6b.
- Scalar distance kernels only; SIMD kernels are Phase 5.

## Benchmarks

Phase 3 results (recall vs `ef_search` at 100K, distance computations per query, build time, HNSW vs Flat
latency, heuristic vs simple selection, inner-product reachability, visited-set strategies) are in
[`benchmarks/results/2026-09-14_ryzen7-4800h_msvc-release_phase3`](../benchmarks/results/2026-09-14_ryzen7-4800h_msvc-release_phase3/README.md).

```powershell
out\build\msvc-release\benchmarks\vf_bench.exe --n 100000 --queries 1000 --dim 128 --metric l2 `
  --ef-search 10,16,32,64,128,256,512 --out hnsw_l2.json
```
