#pragma once

// Hierarchical Navigable Small World backend (docs/DESIGN.md §9, §11.4).
//
// Indexes rows of the collection's VectorStore and reads deletion markers from its TombstoneSet
// (owning neither). Tombstoned nodes stay in the graph as navigation hubs and are filtered from
// query results (§9.9).
//
// Two ways to insert:
//   - add(id): one call with the strong guarantee (Level A, exclusive access).
//   - reserve_node(id) under exclusive access, then link(id), which may run concurrently with
//     other link() calls and with search()/stats() (Level B, docs/concurrency.md). A node is
//     invisible to searches and other inserts until link() publishes it.
//
// Thread safety: search(), search_with(), stats(), build_stats() and link() may run concurrently
// with each other. add(), reserve_node(), prepare_insert(), remove() and set_search_options()
// require exclusive access.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

#include <vectorforge/config.hpp>
#include <vectorforge/status.hpp>
#include <vectorforge/types.hpp>

#include "concurrency/striped_mutex.hpp"
#include "index/hnsw/hnsw_graph.hpp"
#include "index/hnsw/level_generator.hpp"
#include "index/hnsw/neighbor_select.hpp"
#include "index/index_backend.hpp"
#include "index/query_distance.hpp"
#include "search/context_pool.hpp"
#include "search/heaps.hpp"
#include "search/search_context.hpp"
#include "simd/kernels.hpp"
#include "storage/tombstones.hpp"
#include "storage/vector_store.hpp"

namespace vf::detail {

// Internal construction switches (benchmarked alternatives; not part of the public API).
struct HnswBuildOptions {
  NeighborSelection selection = NeighborSelection::Heuristic;
  bool keep_pruned = true;
  // If a new node ends up in no neighbour's list on some level (all back-links pruned), link it
  // from the nearest construction candidate on that level that has a free slot (§9.10).
  bool repair_orphans = true;
  std::size_t nodes_per_chunk = 0;    // HnswGraph::Options
  std::size_t arena_chunk_words = 0;  // HnswGraph::Options
};

// Internal query-path switches (benchmarked alternatives; not part of the public API).
struct HnswSearchOptions {
  // Before computing distances to a node's neighbours, issue a prefetch hint for each unvisited
  // neighbour's vector row. Applies to construction searches too. On by default: +13-16% QPS and
  // -5% build time at d = 768, within run-to-run variation at d = 128 (docs/simd.md "Prefetch").
  bool prefetch = true;
};

struct HnswBuildStats {
  std::uint64_t distance_computations = 0;  // during add(), including neighbour selection
  std::uint64_t orphan_repairs = 0;         // levels on which repair_orphans added a link
  std::uint64_t orphans_unrepaired = 0;     // levels left without an in-link (no free slot found)
};

// Working memory of one insertion. After prepare() for a graph of at most `nodes` nodes, only the
// candidate queue of the neighbour search can still allocate.
struct InsertScratch {
  SearchContext context;
  std::vector<std::vector<ScoredId>> layer_candidates;  // sorted search_layer results per level
  std::vector<std::vector<ScoredId>> layer_selected;    // chosen neighbours per level
  std::vector<ScoredId> shrink_candidates;
  std::vector<ScoredId> shrink_selected;
  std::vector<ScoredId> discarded;
  std::vector<InternalId> id_buffer;
  HnswBuildStats stats;  // of the current insertion

  // Throws std::bad_alloc.
  void prepare(std::size_t nodes, std::uint32_t ef, std::uint32_t max_list, std::uint8_t max_level);
  [[nodiscard]] std::size_t bytes() const noexcept;
};

// Scratch objects for insertions. ensure() runs under exclusive access and creates and sizes
// enough objects that the leases of the following concurrent insertions never allocate.
class InsertScratchPool {
 public:
  class Lease {
   public:
    explicit Lease(InsertScratchPool& pool) noexcept : pool_(&pool), scratch_(pool.take()) {}
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    Lease(Lease&&) = delete;
    Lease& operator=(Lease&&) = delete;
    ~Lease() { pool_->give_back(std::move(scratch_)); }

    [[nodiscard]] InsertScratch& operator*() const noexcept { return *scratch_; }

   private:
    InsertScratchPool* pool_;
    std::unique_ptr<InsertScratch> scratch_;
  };

  // Precondition: nothing is leased. Throws std::bad_alloc (existing objects stay usable).
  void ensure(std::size_t count, std::size_t nodes, std::uint32_t ef, std::uint32_t max_list,
              std::uint8_t max_level);
  [[nodiscard]] std::size_t bytes() const noexcept;

 private:
  // Precondition: fewer leases outstanding than the `count` of the last ensure().
  [[nodiscard]] std::unique_ptr<InsertScratch> take() noexcept;
  void give_back(std::unique_ptr<InsertScratch> scratch) noexcept;

  mutable std::mutex mutex_;
  std::vector<std::unique_ptr<InsertScratch>> free_;
  std::size_t created_ = 0;
};

class HnswBackend final : public IndexBackend {
 public:
  // `vectors` and `deleted` must outlive the backend. Errors: InvalidArgument (params).
  [[nodiscard]] static Result<std::unique_ptr<HnswBackend>> create(
      const VectorStore& vectors, const TombstoneSet& deleted, Metric metric, bool normalized,
      const KernelTable& kernels, const HnswParams& params, const HnswBuildOptions& options = {});

  // Backend over an already built graph (index loading). Preconditions: `graph` passed the reader's
  // structural validation, has graph.m() == params.M and one node per row of `vectors`.
  // Errors: InvalidArgument (params).
  [[nodiscard]] static Result<std::unique_ptr<HnswBackend>> create_loaded(
      const VectorStore& vectors, const TombstoneSet& deleted, Metric metric, bool normalized,
      const KernelTable& kernels, const HnswParams& params, HnswGraph graph);

  [[nodiscard]] IndexType type() const noexcept override { return IndexType::Hnsw; }

  // Inserts row `id` (== graph node count, the last appended row). Errors: ResourceExhausted.
  // Throws std::bad_alloc before the graph is modified; once linking starts nothing allocates.
  [[nodiscard]] Status add(InternalId id) override;

  // Level B, step 0 (exclusive): sizes scratch for `workers` concurrent link() calls on a graph of
  // up to `nodes` nodes. Throws std::bad_alloc (nothing observable changes).
  void prepare_insert(std::uint64_t nodes, std::size_t workers);
  // Level B, step 1 (exclusive): allocates node `id` (== node count, a row already appended) with
  // empty lists; it is not reachable yet. Errors: ResourceExhausted. Throws std::bad_alloc; on any
  // failure the graph is unchanged.
  [[nodiscard]] Status reserve_node(InternalId id);
  // Level B, step 2 (concurrent): connects reserved node `id` and publishes it. Preconditions:
  // prepare_insert() covered the graph size and the number of concurrent calls; each reserved id
  // is linked once. Throws std::bad_alloc only before the node becomes reachable; the node then
  // stays unreachable and the caller must tombstone its row.
  void link(InternalId id);
  void remove(InternalId id) noexcept override;

  [[nodiscard]] std::size_t search(const QueryView& query, const SearchKnobs& knobs,
                                   std::span<Neighbor> out) const override;
  // As search(), using a caller-owned context (its distance counter is incremented).
  [[nodiscard]] std::size_t search_with(SearchContext& context, const QueryView& query,
                                        const SearchKnobs& knobs, std::span<Neighbor> out) const;

  [[nodiscard]] BackendStats stats() const noexcept override;

  [[nodiscard]] const HnswGraph& graph() const noexcept { return graph_; }
  [[nodiscard]] const HnswParams& params() const noexcept { return params_; }
  [[nodiscard]] HnswBuildStats build_stats() const noexcept;
  [[nodiscard]] const HnswSearchOptions& search_options() const noexcept { return search_options_; }
  // Requires exclusive access (no concurrent search or add).
  void set_search_options(const HnswSearchOptions& options) noexcept { search_options_ = options; }

 private:
  HnswBackend(const VectorStore& vectors, const TombstoneSet& deleted, Metric metric,
              bool normalized, const KernelTable& kernels, const HnswParams& params,
              const HnswBuildOptions& options, HnswGraph graph);

  [[nodiscard]] float distance(SearchContext& context, const QueryView& query,
                               InternalId id) const noexcept {
    ++context.distance_computations;
    return query_distance(*kernels_, mode_, query, vectors_->row_ptr(id), dim_);
  }
  // Distance between two stored rows (construction only).
  [[nodiscard]] float node_distance(InsertScratch& scratch, InternalId a,
                                    InternalId b) const noexcept {
    ++scratch.stats.distance_computations;
    return query_distance(*kernels_, mode_, QueryView{.data = vectors_->row_ptr(a), .inv_norm = 1},
                          vectors_->row_ptr(b), dim_);
  }

  // Greedy descent with beam 1 on `level` (§9.5).
  [[nodiscard]] ScoredId greedy_search(SearchContext& context, const QueryView& query,
                                       ScoredId current, std::uint8_t level) const noexcept;
  // Beam search on `level` (paper Algorithm 2). Results live in context.results.
  [[nodiscard]] BoundedMaxHeap<ScoredId> search_layer(SearchContext& context,
                                                      const QueryView& query,
                                                      std::span<const ScoredId> entry_points,
                                                      std::uint32_t ef, std::uint8_t level,
                                                      bool filter_deleted) const;

  // Insertion phase 1: candidates and selections for levels [0, top] in `scratch`. Reads only
  // published nodes. Throws std::bad_alloc.
  void find_neighbors(InsertScratch& scratch, InternalId id, std::uint8_t top,
                      HnswGraph::Entry entry) const;
  // Insertion phases 2 and 3: writes the lists of `id`, then the back-links that publish it.
  void publish(InsertScratch& scratch, InternalId id, std::uint8_t top) noexcept;
  // Adds `new_id` to the full list of `node` on `level` by re-selecting; returns whether it was
  // kept. The caller holds the stripe of `node`.
  bool shrink_with(InsertScratch& scratch, InternalId node, std::uint8_t level, InternalId new_id,
                   float new_distance) noexcept;
  void write_list(InsertScratch& scratch, InternalId node, std::uint8_t level,
                  std::span<const ScoredId> selection) noexcept;
  // Adds the statistics of a finished insertion.
  void record(const InsertScratch& scratch) noexcept;

  static constexpr std::size_t kLinkStripes = 4096;

  const VectorStore* vectors_;
  const TombstoneSet* deleted_;
  const KernelTable* kernels_;
  std::size_t dim_;
  ScoreMode mode_;
  HnswParams params_;
  HnswBuildOptions options_;
  HnswSearchOptions search_options_;
  LevelGenerator levels_;
  HnswGraph graph_;

  mutable ContextPool contexts_;
  InsertScratchPool scratch_;
  // Writers of a node's lists hold link_locks_.for_key(node); an insertion that may raise the top
  // level holds top_mutex_ for its whole duration (docs/concurrency.md, "Level B").
  StripedMutex link_locks_{kLinkStripes};
  std::mutex top_mutex_;

  std::atomic<std::uint64_t> distance_computations_{0};
  std::atomic<std::uint64_t> orphan_repairs_{0};
  std::atomic<std::uint64_t> orphans_unrepaired_{0};
};

}  // namespace vf::detail
