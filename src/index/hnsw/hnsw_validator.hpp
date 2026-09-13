#pragma once

// Structural checks for an HnswGraph (docs/DESIGN.md §9.3, §9.10). Used by tests after every
// integration build and, from Phase 4, by `vectorforge verify`.

#include <cstdint>
#include <vector>

#include <vectorforge/status.hpp>

#include "index/hnsw/hnsw_graph.hpp"

namespace vf::detail {

struct HnswReachability {
  std::uint64_t nodes = 0;
  // unreachable[l] = nodes with level >= l not reachable on level l from the entry point, following
  // only level-l links. Level 0 covers every node. Size: entry level + 1 (empty for an empty
  // graph).
  std::vector<std::uint64_t> unreachable;

  [[nodiscard]] std::uint64_t unreachable_level0() const noexcept {
    return unreachable.empty() ? 0 : unreachable[0];
  }
};

class HnswValidator {
 public:
  explicit HnswValidator(const HnswGraph& graph) noexcept : graph_(&graph) {}

  // Checks, for every list: count <= capacity(level); every id < node_count; no self-links; no
  // duplicates; linked nodes have level >= list level. And: the entry point is valid iff the graph
  // is non-empty and has the maximum level of all nodes. Returns the first violation found as
  // Internal.
  [[nodiscard]] Status check_invariants() const;

  // Breadth-first search per level from the entry point. O(nodes + links).
  // Precondition: check_invariants().ok().
  [[nodiscard]] HnswReachability reachability() const;

  // histogram[l] = number of nodes whose top level is l (size max_level + 1).
  [[nodiscard]] std::vector<std::uint64_t> level_histogram() const;

 private:
  const HnswGraph* graph_;
};

}  // namespace vf::detail
