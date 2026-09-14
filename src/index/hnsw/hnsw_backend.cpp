#include "index/hnsw/hnsw_backend.hpp"

#include <utility>

namespace vf::detail {

Result<std::unique_ptr<HnswBackend>> HnswBackend::create(
    const VectorStore& vectors, const TombstoneSet& deleted, Metric metric, bool normalized,
    const KernelTable& kernels, const HnswParams& params, const HnswBuildOptions& options) {
  VF_RETURN_IF_ERROR(params.validate());
  Result<HnswGraph> graph = HnswGraph::create({.m = params.M,
                                               .max_level = params.max_level,
                                               .nodes_per_chunk = options.nodes_per_chunk,
                                               .arena_chunk_words = options.arena_chunk_words});
  if (!graph.ok()) {
    return graph.status();
  }
  return std::unique_ptr<HnswBackend>(new HnswBackend(vectors, deleted, metric, normalized, kernels,
                                                      params, options, std::move(graph).value()));
}

Result<std::unique_ptr<HnswBackend>> HnswBackend::create_loaded(
    const VectorStore& vectors, const TombstoneSet& deleted, Metric metric, bool normalized,
    const KernelTable& kernels, const HnswParams& params, HnswGraph graph) {
  VF_RETURN_IF_ERROR(params.validate());
  VF_CHECK(graph.m() == params.M && graph.node_count() == vectors.size(),
           "HnswBackend::create_loaded: graph does not match parameters or vectors");
  return std::unique_ptr<HnswBackend>(new HnswBackend(
      vectors, deleted, metric, normalized, kernels, params, HnswBuildOptions{}, std::move(graph)));
}

HnswBackend::HnswBackend(const VectorStore& vectors, const TombstoneSet& deleted, Metric metric,
                         bool normalized, const KernelTable& kernels, const HnswParams& params,
                         const HnswBuildOptions& options, HnswGraph graph)
    : vectors_(&vectors),
      deleted_(&deleted),
      kernels_(&kernels),
      dim_(vectors.dim()),
      mode_(score_mode(metric, normalized)),
      params_(params),
      options_(options),
      levels_(params.seed, params.M, params.max_level),
      graph_(std::move(graph)) {
  // Scratch used after the graph has been modified must never allocate (see add()).
  const std::size_t max_list = static_cast<std::size_t>(graph_.capacity(0)) + 1;
  shrink_candidates_.reserve(max_list);
  shrink_selected_.reserve(max_list);
  discarded_.reserve(max_list);
  id_buffer_.reserve(max_list);
}

void HnswBackend::remove(InternalId /*id*/) noexcept {
  // Tombstones are read from the collection's TombstoneSet; the node stays as a navigation hub.
}

std::size_t HnswBackend::search(const QueryView& query, const SearchKnobs& knobs,
                                std::span<Neighbor> out) const {
  const ContextPool::Lease context = contexts_.acquire();
  return search_with(*context, query, knobs, out);
}

BackendStats HnswBackend::stats() const noexcept {
  std::size_t scratch = build_context_.bytes();
  for (const auto& level : layer_candidates_) {
    scratch += level.capacity() * sizeof(ScoredId);
  }
  for (const auto& level : layer_selected_) {
    scratch += level.capacity() * sizeof(ScoredId);
  }
  return {.index_bytes = graph_.bytes() + scratch};
}

}  // namespace vf::detail
