#pragma once

// Per-search scratch state for graph search (docs/DESIGN.md §9.5): visited set, candidate queue,
// result storage and a distance-computation counter. Contexts are reused across searches, so after
// warm-up a search allocates nothing unless the graph or the beam width grew.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "search/heaps.hpp"
#include "search/visited_set.hpp"

namespace vf::detail {

struct SearchContext {
  VisitedSet visited;
  MinHeap<ScoredId> candidates;
  std::vector<ScoredId> results;  // storage for a BoundedMaxHeap of the beam width
  std::uint64_t distance_computations = 0;

  // Sizes the context for a graph of `nodes` nodes and beam width `ef`. Throws std::bad_alloc.
  void prepare(std::size_t nodes, std::size_t ef) {
    visited.ensure_capacity(nodes);
    if (results.size() < ef) {
      results.resize(ef);
    }
  }

  [[nodiscard]] std::size_t bytes() const noexcept {
    return visited.bytes() + (results.capacity() * sizeof(ScoredId)) +
           (candidates.capacity() * sizeof(ScoredId));
  }
};

}  // namespace vf::detail
