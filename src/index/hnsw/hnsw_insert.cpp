// HNSW insertion with explicit publication order (docs/DESIGN.md §9.6, §9.7, §9.10, §11.4).

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>

#include "core/assert.hpp"
#include "index/hnsw/hnsw_backend.hpp"

namespace vf::detail {

void HnswBackend::write_list(InsertScratch& scratch, InternalId node, std::uint8_t level,
                             std::span<const ScoredId> selection) noexcept {
  scratch.id_buffer.clear();
  for (const ScoredId& s : selection) {
    scratch.id_buffer.push_back(s.id);  // capacity reserved by InsertScratch::prepare
  }
  graph_.set_links(node, level, scratch.id_buffer);
}

bool HnswBackend::shrink_with(InsertScratch& scratch, InternalId node, std::uint8_t level,
                              InternalId new_id, float new_distance) noexcept {
  // All scratch vectors have capacity >= capacity(0) + 1 (prepare), so nothing allocates.
  std::vector<ScoredId>& candidates = scratch.shrink_candidates;
  candidates.clear();
  const LinkView links = graph_.links(node, level);
  for (std::uint32_t i = 0; i < links.size(); ++i) {
    const InternalId neighbor = links[i];
    candidates.push_back({.distance = node_distance(scratch, node, neighbor), .id = neighbor});
  }
  // dist(new, node) == dist(node, new) bit for bit (index/query_distance.hpp), so reuse it.
  candidates.push_back({.distance = new_distance, .id = new_id});
  // Insertion sort of at most 2M + 1 elements. Unlike std::sort it stays memory-safe if a distance
  // is NaN (possible only for vector data loaded without checksum verification), where the
  // comparison is not a strict weak ordering.
  for (std::size_t i = 1; i < candidates.size(); ++i) {
    const ScoredId key = candidates[i];
    std::size_t j = i;
    for (; j > 0 && candidate_less(key, candidates[j - 1]); --j) {
      candidates[j] = candidates[j - 1];
    }
    candidates[j] = key;
  }
  select_neighbors(
      candidates, graph_.capacity(level), options_.selection, options_.keep_pruned,
      [this, &scratch](InternalId a, InternalId b) { return node_distance(scratch, a, b); },
      scratch.shrink_selected, scratch.discarded);
  write_list(scratch, node, level, scratch.shrink_selected);
  return std::any_of(scratch.shrink_selected.begin(), scratch.shrink_selected.end(),
                     [new_id](const ScoredId& s) { return s.id == new_id; });
}

void HnswBackend::find_neighbors(InsertScratch& scratch, InternalId id, std::uint8_t top,
                                 HnswGraph::Entry entry) const {
  SearchContext& context = scratch.context;
  const QueryView query{.data = vectors_->row_ptr(id), .inv_norm = 1.0F};
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
    std::vector<ScoredId>& candidates = scratch.layer_candidates[lvl];
    candidates.assign(context.results.begin(),
                      context.results.begin() + static_cast<std::ptrdiff_t>(count));
    select_neighbors(
        candidates, params_.M, options_.selection, options_.keep_pruned,
        [this, &scratch](InternalId a, InternalId b) { return node_distance(scratch, a, b); },
        scratch.layer_selected[lvl], scratch.discarded);
    entry_points = candidates;  // the next level starts from all of W (paper Algorithm 1)
  }
}

void HnswBackend::publish(InsertScratch& scratch, InternalId id, std::uint8_t top) noexcept {
  // Phase 2: the node's own lists. No other thread can reach the node yet, so no lock is needed.
  for (std::uint8_t l = 0; l <= top; ++l) {
    write_list(scratch, id, l, scratch.layer_selected[l]);
  }

  // Phase 3: back-links, each under the stripe of the list being changed. The first one makes the
  // node reachable.
  for (std::uint8_t l = 0; l <= top; ++l) {
    std::size_t in_links = 0;
    for (const ScoredId& neighbor : scratch.layer_selected[l]) {
      const std::lock_guard<std::mutex> lock(link_locks_.for_key(neighbor.id));
      if (graph_.try_append_link(neighbor.id, l, id) ||
          shrink_with(scratch, neighbor.id, l, id, neighbor.distance)) {
        ++in_links;
      }
    }
    if (in_links == 0 && options_.repair_orphans) {
      // Every neighbour pruned the new node (typical for duplicates). Link it from the nearest
      // construction candidate that still has room; candidates all have level >= l.
      const bool repaired =
          std::any_of(scratch.layer_candidates[l].begin(), scratch.layer_candidates[l].end(),
                      [&](const ScoredId& c) {
                        const std::lock_guard<std::mutex> lock(link_locks_.for_key(c.id));
                        return graph_.try_append_link(c.id, l, id);
                      });
      if (repaired) {
        ++scratch.stats.orphan_repairs;
      } else {
        ++scratch.stats.orphans_unrepaired;
      }
    } else if (in_links == 0) {
      ++scratch.stats.orphans_unrepaired;
    }
  }
}

void HnswBackend::record(const InsertScratch& scratch) noexcept {
  distance_computations_.fetch_add(
      scratch.stats.distance_computations + scratch.context.distance_computations,
      std::memory_order_relaxed);
  orphan_repairs_.fetch_add(scratch.stats.orphan_repairs, std::memory_order_relaxed);
  orphans_unrepaired_.fetch_add(scratch.stats.orphans_unrepaired, std::memory_order_relaxed);
}

void HnswBackend::prepare_insert(std::uint64_t nodes, std::size_t workers) {
  scratch_.ensure(workers, static_cast<std::size_t>(nodes), params_.ef_construction,
                  graph_.capacity(0), graph_.max_level());
}

Status HnswBackend::reserve_node(InternalId id) {
  VF_ASSERT(id == graph_.node_count() && static_cast<std::uint64_t>(id) < vectors_->size(),
            "HnswBackend::reserve_node: row must be the next graph node");
  return graph_.add_node(levels_.level_for(id));
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

  // Phase 1: pure reads; everything that allocates happens before the graph changes.
  prepare_insert(graph_.node_count(), 1);
  const InsertScratchPool::Lease lease(scratch_);
  InsertScratch& scratch = *lease;
  scratch.stats = {};
  scratch.context.distance_computations = 0;
  const std::uint8_t top = std::min(level, entry.level);
  find_neighbors(scratch, id, top, entry);

  VF_RETURN_IF_ERROR(graph_.add_node(level));
  publish(scratch, id, top);
  // Phase 4: a node above the current top level becomes the entry point.
  if (level > entry.level) {
    graph_.set_entry({.id = id, .level = level});
  }
  record(scratch);
  return {};
}

void HnswBackend::link(InternalId id) {
  const std::uint8_t level = graph_.level(id);
  // Only an insertion above the current top level can change the entry point. It holds top_mutex_
  // throughout, so two such insertions cannot both link below a level neither of them has joined.
  std::unique_lock<std::mutex> top_lock(top_mutex_, std::defer_lock);
  HnswGraph::Entry entry = graph_.entry();
  if (!entry.valid() || level > entry.level) {
    top_lock.lock();
    entry = graph_.entry();
    if (!entry.valid()) {
      graph_.set_entry({.id = id, .level = level});
      return;
    }
  }

  const InsertScratchPool::Lease lease(scratch_);
  InsertScratch& scratch = *lease;
  scratch.stats = {};
  scratch.context.distance_computations = 0;
  const std::uint8_t top = std::min(level, entry.level);
  find_neighbors(scratch, id, top, entry);
  publish(scratch, id, top);
  if (level > entry.level) {
    graph_.set_entry({.id = id, .level = level});
  }
  record(scratch);
}

}  // namespace vf::detail
