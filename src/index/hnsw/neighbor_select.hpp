#pragma once

// HNSW neighbour selection (docs/DESIGN.md §9.7; paper Algorithm 4 without extendCandidates, and
// Algorithm 3 as the `Simple` variant).

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "search/heaps.hpp"

namespace vf::detail {

enum class NeighborSelection : std::uint8_t {
  // Keep a candidate only if it is closer to the base than to every already selected neighbour.
  Heuristic,
  // Keep the m closest candidates.
  Simple,
};

// Selects at most `m` neighbours for a base node from `sorted`, which holds candidates sorted
// ascending by (distance to base, id) with distinct ids, none equal to the base.
// `distance(a, b)` returns the distance between candidate nodes a and b (called only by the
// heuristic). With keep_pruned, the heuristic fills remaining slots with pruned candidates in
// ascending order. The selection is written to `out` (cleared first): accepted candidates in
// ascending order, followed by any refilled pruned ones. `discarded` is scratch. Neither vector
// allocates when its capacity is at least sorted.size().
template <class DistanceFn>
void select_neighbors(std::span<const ScoredId> sorted, std::size_t m, NeighborSelection selection,
                      bool keep_pruned, DistanceFn&& distance, std::vector<ScoredId>& out,
                      std::vector<ScoredId>& discarded) {
  out.clear();
  discarded.clear();
  if (selection == NeighborSelection::Simple) {
    const std::size_t n = sorted.size() < m ? sorted.size() : m;
    out.assign(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(n));
    return;
  }
  for (const ScoredId& candidate : sorted) {
    if (out.size() == m) {
      break;
    }
    bool good = true;
    for (const ScoredId& selected : out) {
      if (distance(candidate.id, selected.id) < candidate.distance) {
        good = false;
        break;
      }
    }
    if (good) {
      out.push_back(candidate);
    } else if (keep_pruned) {
      discarded.push_back(candidate);
    }
  }
  if (keep_pruned) {
    for (const ScoredId& candidate : discarded) {
      if (out.size() == m) {
        break;
      }
      out.push_back(candidate);
    }
  }
}

}  // namespace vf::detail
