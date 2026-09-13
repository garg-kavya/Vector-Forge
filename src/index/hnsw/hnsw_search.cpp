// HNSW search primitives and the query path (docs/DESIGN.md §9.5).

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>

#include "core/assert.hpp"
#include "index/hnsw/hnsw_backend.hpp"

namespace vf::detail {

ScoredId HnswBackend::greedy_search(SearchContext& context, const QueryView& query,
                                    ScoredId current, std::uint8_t level) const noexcept {
  // (distance, id) strictly decreases with every move, so the loop terminates without a visited
  // set.
  for (bool changed = true; changed;) {
    changed = false;
    const LinkView links = graph_.links(current.id, level);
    for (std::uint32_t i = 0; i < links.size(); ++i) {
      const InternalId neighbor = links[i];
      const ScoredId candidate{.distance = distance(context, query, neighbor), .id = neighbor};
      if (candidate_less(candidate, current)) {
        current = candidate;
        changed = true;
      }
    }
  }
  return current;
}

BoundedMaxHeap<ScoredId> HnswBackend::search_layer(SearchContext& context, const QueryView& query,
                                                   std::span<const ScoredId> entry_points,
                                                   std::uint32_t ef, std::uint8_t level,
                                                   bool filter_deleted) const {
  VF_ASSERT(ef >= 1 && context.results.size() >= ef, "search_layer: context not prepared");
  context.visited.reset();
  context.candidates.clear();
  BoundedMaxHeap<ScoredId> results(std::span<ScoredId>(context.results).first(ef));
  const bool check_deleted = filter_deleted && deleted_->any();

  for (const ScoredId& entry : entry_points) {
    if (!context.visited.visit(entry.id)) {
      continue;
    }
    context.candidates.push(entry);
    if (!check_deleted || !deleted_->test(entry.id)) {
      results.push(entry);
    }
  }

  while (!context.candidates.empty()) {
    const ScoredId nearest = context.candidates.top();
    // The closest unexpanded candidate is worse than the worst result: nothing left can improve W.
    if (results.full() && candidate_less(results.top(), nearest)) {
      break;
    }
    context.candidates.pop();
    const LinkView links = graph_.links(nearest.id, level);
    for (std::uint32_t i = 0; i < links.size(); ++i) {
      const InternalId neighbor = links[i];
      if (!context.visited.visit(neighbor)) {
        continue;
      }
      const ScoredId candidate{.distance = distance(context, query, neighbor), .id = neighbor};
      if (!results.full() || candidate_less(candidate, results.top())) {
        context.candidates.push(candidate);
        if (!check_deleted || !deleted_->test(neighbor)) {
          results.push(candidate);
        }
      }
    }
  }
  return results;
}

std::size_t HnswBackend::search_with(SearchContext& context, const QueryView& query,
                                     const SearchKnobs& knobs, std::span<Neighbor> out) const {
  VF_ASSERT(knobs.k >= 1 && out.size() >= knobs.k, "HnswBackend::search: output too small");
  const HnswGraph::Entry entry = graph_.entry();
  if (!entry.valid()) {
    return 0;
  }
  const std::uint32_t ef = std::max(knobs.ef != 0 ? knobs.ef : params_.ef_search, knobs.k);
  context.prepare(static_cast<std::size_t>(graph_.node_count()), ef);

  ScoredId current{.distance = distance(context, query, entry.id), .id = entry.id};
  for (std::uint8_t level = entry.level; level > 0; --level) {
    current = greedy_search(context, query, current, level);
  }
  BoundedMaxHeap<ScoredId> results =
      search_layer(context, query, std::span<const ScoredId>(&current, 1), ef, 0, true);
  const std::size_t found = results.sort_ascending();
  const std::size_t count = std::min<std::size_t>(found, knobs.k);
  for (std::size_t i = 0; i < count; ++i) {
    out[i] = Neighbor{.id = context.results[i].id, .distance = context.results[i].distance};
  }
  return count;
}

}  // namespace vf::detail
