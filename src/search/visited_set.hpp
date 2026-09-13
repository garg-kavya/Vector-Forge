#pragma once

// Epoch-stamped visited set for graph search (docs/DESIGN.md §9.5).
//
// mark[i] == epoch means "visited in the current search". reset() starts a new search in O(1) by
// incrementing the epoch; when the 16-bit epoch wraps around, the array is cleared once and the
// epoch restarts at 1 (0 is never a live epoch, so freshly grown entries are unvisited).
//
// Memory: 2 bytes per node. Thread safety: none (one set per search context).

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <vectorforge/types.hpp>

#include "core/assert.hpp"

namespace vf::detail {

class VisitedSet {
 public:
  // Ensures ids [0, capacity) can be visited. Never shrinks. Throws std::bad_alloc.
  void ensure_capacity(std::size_t capacity) {
    if (capacity > marks_.size()) {
      marks_.resize(capacity, 0);
    }
  }

  // Starts a new search: every id becomes unvisited.
  void reset() noexcept {
    if (epoch_ == std::numeric_limits<std::uint16_t>::max()) {
      std::fill(marks_.begin(), marks_.end(), std::uint16_t{0});
      epoch_ = 1;
      return;
    }
    ++epoch_;
  }

  // Marks `id` visited; returns true if it was not visited before in this search.
  // Precondition: id < capacity().
  bool visit(InternalId id) noexcept {
    VF_ASSERT(id < marks_.size(), "VisitedSet::visit: id out of range");
    // Branch-free: the caller branches on the result anyway, and an unconditional store avoids a
    // second unpredictable branch.
    const std::uint16_t epoch = epoch_;
    std::uint16_t& mark = marks_[id];
    const bool fresh = mark != epoch;
    mark = epoch;
    return fresh;
  }

  [[nodiscard]] bool visited(InternalId id) const noexcept {
    return id < marks_.size() && marks_[id] == epoch_;
  }

  [[nodiscard]] std::size_t capacity() const noexcept { return marks_.size(); }
  [[nodiscard]] std::uint16_t epoch() const noexcept { return epoch_; }
  [[nodiscard]] std::size_t bytes() const noexcept {
    return marks_.capacity() * sizeof(std::uint16_t);
  }

 private:
  std::vector<std::uint16_t> marks_;
  std::uint16_t epoch_ = 1;
};

}  // namespace vf::detail
