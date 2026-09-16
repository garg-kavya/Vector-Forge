#include "index/hnsw/hnsw_backend.hpp"

#include <algorithm>
#include <utility>

namespace vf::detail {

void InsertScratch::prepare(std::size_t nodes, std::uint32_t ef, std::uint32_t max_list,
                            std::uint8_t max_level) {
  context.prepare(nodes, ef);
  const std::size_t levels = static_cast<std::size_t>(max_level) + 1;
  if (layer_candidates.size() < levels) {
    layer_candidates.resize(levels);
  }
  if (layer_selected.size() < levels) {
    layer_selected.resize(levels);
  }
  for (std::size_t l = 0; l < levels; ++l) {
    layer_candidates[l].reserve(ef);
    layer_selected[l].reserve(max_list);
  }
  const std::size_t list = static_cast<std::size_t>(max_list) + 1;
  shrink_candidates.reserve(list);
  shrink_selected.reserve(list);
  discarded.reserve(std::max<std::size_t>(list, ef));
  id_buffer.reserve(list);
}

std::size_t InsertScratch::bytes() const noexcept {
  std::size_t total = context.bytes();
  for (const auto& level : layer_candidates) {
    total += level.capacity() * sizeof(ScoredId);
  }
  for (const auto& level : layer_selected) {
    total += level.capacity() * sizeof(ScoredId);
  }
  total += (shrink_candidates.capacity() + shrink_selected.capacity() + discarded.capacity()) *
           sizeof(ScoredId);
  return total + (id_buffer.capacity() * sizeof(InternalId));
}

void InsertScratchPool::ensure(std::size_t count, std::size_t nodes, std::uint32_t ef,
                               std::uint32_t max_list, std::uint8_t max_level) {
  const std::lock_guard<std::mutex> lock(mutex_);
  VF_ASSERT(free_.size() == created_, "InsertScratchPool::ensure while scratch is leased");
  free_.reserve(count);
  while (created_ < count) {
    free_.push_back(std::make_unique<InsertScratch>());
    ++created_;
  }
  for (const std::unique_ptr<InsertScratch>& scratch : free_) {
    scratch->prepare(nodes, ef, max_list, max_level);
  }
}

std::unique_ptr<InsertScratch> InsertScratchPool::take() noexcept {
  const std::lock_guard<std::mutex> lock(mutex_);
  VF_CHECK(!free_.empty(), "InsertScratchPool: more concurrent insertions than prepared");
  std::unique_ptr<InsertScratch> scratch = std::move(free_.back());
  free_.pop_back();
  return scratch;
}

void InsertScratchPool::give_back(std::unique_ptr<InsertScratch> scratch) noexcept {
  const std::lock_guard<std::mutex> lock(mutex_);
  free_.push_back(std::move(scratch));  // capacity >= created_ (ensure)
}

std::size_t InsertScratchPool::bytes() const noexcept {
  const std::lock_guard<std::mutex> lock(mutex_);
  std::size_t total = 0;
  for (const std::unique_ptr<InsertScratch>& scratch : free_) {
    total += scratch->bytes();
  }
  return total;
}

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
}

void HnswBackend::remove(InternalId /*id*/) noexcept {
  // Tombstones are read from the collection's TombstoneSet; the node stays as a navigation hub.
}

std::size_t HnswBackend::search(const QueryView& query, const SearchKnobs& knobs,
                                std::span<Neighbor> out) const {
  const ContextPool::Lease context = contexts_.acquire();
  return search_with(*context, query, knobs, out);
}

HnswBuildStats HnswBackend::build_stats() const noexcept {
  return {.distance_computations = distance_computations_.load(),
          .orphan_repairs = orphan_repairs_.load(),
          .orphans_unrepaired = orphans_unrepaired_.load()};
}

BackendStats HnswBackend::stats() const noexcept {
  return {.index_bytes = graph_.bytes() + scratch_.bytes()};
}

}  // namespace vf::detail
