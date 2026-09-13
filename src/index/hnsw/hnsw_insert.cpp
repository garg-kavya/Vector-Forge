// HNSW insertion with explicit publication order (docs/DESIGN.md §9.6, §9.7, §9.10).

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>

#include "core/assert.hpp"
#include "index/hnsw/hnsw_backend.hpp"

namespace vf::detail {

void HnswBackend::write_list(InternalId node, std::uint8_t level,
                             std::span<const ScoredId> selection) noexcept {
  id_buffer_.clear();
  for (const ScoredId& s : selection) {
    id_buffer_.push_back(s.id);  // capacity reserved in the constructor
  }
  graph_.set_links(node, level, id_buffer_);
}

bool HnswBackend::shrink_with(InternalId node, std::uint8_t level, InternalId new_id,
                              float new_distance) noexcept {
  // All scratch vectors have capacity >= capacity(0) + 1 (constructor), so nothing allocates.
  shrink_candidates_.clear();
  const LinkView links = graph_.links(node, level);
  for (std::uint32_t i = 0; i < links.size(); ++i) {
    const InternalId neighbor = links[i];
    shrink_candidates_.push_back({.distance = node_distance(node, neighbor), .id = neighbor});
  }
  // dist(new, node) == dist(node, new) bit for bit (index/query_distance.hpp), so reuse it.
  shrink_candidates_.push_back({.distance = new_distance, .id = new_id});
  std::sort(shrink_candidates_.begin(), shrink_candidates_.end(),
            [](const ScoredId& a, const ScoredId& b) { return candidate_less(a, b); });
  select_neighbors(
      shrink_candidates_, graph_.capacity(level), options_.selection, options_.keep_pruned,
      [this](InternalId a, InternalId b) { return node_distance(a, b); }, shrink_selected_,
      discarded_);
  write_list(node, level, shrink_selected_);
  return std::any_of(shrink_selected_.begin(), shrink_selected_.end(),
                     [new_id](const ScoredId& s) { return s.id == new_id; });
}

Status HnswBackend::add(InternalId id) {
  VF_ASSERT(id == graph_.node_count() && static_cast<std::uint64_t>(id) + 1 == vectors_->size(),
            "HnswBackend::add: row must be the next graph node and the last appended row");
  const std::uint8_t level = levels_.level_for(id);
  const HnswGraph::Entry entry = graph_.entry();
  if (!entry.valid()) {
    VF_RETURN_IF_ERROR(graph_.add_node(level));
    graph_.set_entry({.id = id, .level = level});
    return {};
  }

  // Phase 1: pure reads. Compute neighbour candidates and selections for every level the node
  // shares with the graph. Everything that allocates happens here, before the graph changes.
  const QueryView query{.data = vectors_->row_ptr(id), .inv_norm = 1.0F};
  SearchContext& context = build_context_;
  context.prepare(static_cast<std::size_t>(graph_.node_count()), params_.ef_construction);
  const std::uint8_t top = std::min(level, entry.level);
  // Sized independently: if one resize throws, the next add() must not assume the other grew.
  if (layer_candidates_.size() <= top) {
    layer_candidates_.resize(static_cast<std::size_t>(top) + 1);
  }
  if (layer_selected_.size() <= top) {
    layer_selected_.resize(static_cast<std::size_t>(top) + 1);
  }

  const std::uint64_t search_distances_before = context.distance_computations;
  ScoredId current{.distance = distance(context, query, entry.id), .id = entry.id};
  for (std::uint8_t l = entry.level; l > top; --l) {
    current = greedy_search(context, query, current, l);
  }
  std::span<const ScoredId> entry_points(&current, 1);
  for (int l = top; l >= 0; --l) {
    const auto lvl = static_cast<std::uint8_t>(l);
    BoundedMaxHeap<ScoredId> found =
        search_layer(context, query, entry_points, params_.ef_construction, lvl, false);
    const std::size_t count = found.sort_ascending();
    std::vector<ScoredId>& candidates = layer_candidates_[lvl];
    candidates.assign(context.results.begin(),
                      context.results.begin() + static_cast<std::ptrdiff_t>(count));
    if (discarded_.capacity() < count) {
      discarded_.reserve(count);
    }
    layer_selected_[lvl].reserve(params_.M);
    select_neighbors(
        candidates, params_.M, options_.selection, options_.keep_pruned,
        [this](InternalId a, InternalId b) { return node_distance(a, b); }, layer_selected_[lvl],
        discarded_);
    entry_points = candidates;  // the next level starts from all of W (paper Algorithm 1)
  }
  build_stats_.distance_computations += context.distance_computations - search_distances_before;

  VF_RETURN_IF_ERROR(graph_.add_node(level));

  // Phase 2: write the new node's own lists. It is not yet reachable from any other node.
  for (std::uint8_t l = 0; l <= top; ++l) {
    write_list(id, l, layer_selected_[l]);
  }

  // Phase 3: back-links. The node becomes reachable here.
  for (std::uint8_t l = 0; l <= top; ++l) {
    std::size_t in_links = 0;
    for (const ScoredId& neighbor : layer_selected_[l]) {
      if (graph_.try_append_link(neighbor.id, l, id) ||
          shrink_with(neighbor.id, l, id, neighbor.distance)) {
        ++in_links;
      }
    }
    if (in_links == 0 && options_.repair_orphans) {
      // Every neighbour pruned the new node (typical for duplicates). Link it from the nearest
      // construction candidate that still has room; candidates all have level >= l.
      const bool repaired =
          std::any_of(layer_candidates_[l].begin(), layer_candidates_[l].end(),
                      [&](const ScoredId& c) { return graph_.try_append_link(c.id, l, id); });
      if (repaired) {
        ++build_stats_.orphan_repairs;
      } else {
        ++build_stats_.orphans_unrepaired;
      }
    } else if (in_links == 0) {
      ++build_stats_.orphans_unrepaired;
    }
  }

  // Phase 4: a node above the current top level becomes the entry point.
  if (level > entry.level) {
    graph_.set_entry({.id = id, .level = level});
  }
  return {};
}

}  // namespace vf::detail
