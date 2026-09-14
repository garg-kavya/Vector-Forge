#pragma once

// Hierarchical Navigable Small World backend (docs/DESIGN.md §9), single-threaded build.
//
// Indexes rows of the collection's VectorStore and reads deletion markers from its TombstoneSet
// (owning neither). Tombstoned nodes stay in the graph as navigation hubs and are filtered from
// query results (§9.9).
//
// Thread safety (Phase 3): search() and stats() may run concurrently with each other (each search
// leases its own SearchContext from a mutex-protected pool); add() and remove() require exclusive
// access.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <vectorforge/config.hpp>
#include <vectorforge/status.hpp>
#include <vectorforge/types.hpp>

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
  void remove(InternalId id) noexcept override;

  [[nodiscard]] std::size_t search(const QueryView& query, const SearchKnobs& knobs,
                                   std::span<Neighbor> out) const override;
  // As search(), using a caller-owned context (its distance counter is incremented).
  [[nodiscard]] std::size_t search_with(SearchContext& context, const QueryView& query,
                                        const SearchKnobs& knobs, std::span<Neighbor> out) const;

  [[nodiscard]] BackendStats stats() const noexcept override;

  [[nodiscard]] const HnswGraph& graph() const noexcept { return graph_; }
  [[nodiscard]] const HnswParams& params() const noexcept { return params_; }
  [[nodiscard]] const HnswBuildStats& build_stats() const noexcept { return build_stats_; }
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
  [[nodiscard]] float node_distance(InternalId a, InternalId b) noexcept {
    ++build_stats_.distance_computations;
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
  // Adds `new_id` to the full list of `node` on `level` by re-selecting; returns whether it was
  // kept.
  bool shrink_with(InternalId node, std::uint8_t level, InternalId new_id,
                   float new_distance) noexcept;
  void write_list(InternalId node, std::uint8_t level,
                  std::span<const ScoredId> selection) noexcept;

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
  HnswBuildStats build_stats_;

  mutable ContextPool contexts_;

  // Construction scratch, reused across add() calls.
  SearchContext build_context_;
  std::vector<std::vector<ScoredId>> layer_candidates_;  // sorted search_layer results per level
  std::vector<std::vector<ScoredId>> layer_selected_;    // chosen neighbours per level
  std::vector<ScoredId> shrink_candidates_;
  std::vector<ScoredId> shrink_selected_;
  std::vector<ScoredId> discarded_;
  std::vector<InternalId> id_buffer_;
};

}  // namespace vf::detail
