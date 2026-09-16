#pragma once

// Fixed array of mutexes indexed by key (docs/DESIGN.md §11.4). Two keys may share a mutex, which
// only serialises them. Callers hold at most one stripe at a time, so stripes cannot deadlock.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

#include "core/assert.hpp"

namespace vf::detail {

class StripedMutex {
 public:
  // `stripes` must be a power of two. Throws std::bad_alloc.
  explicit StripedMutex(std::size_t stripes)
      : mask_(stripes - 1),
        mutexes_(std::make_unique<std::mutex[]>(stripes)) {  // NOLINT(modernize-avoid-c-arrays)
    VF_CHECK(stripes != 0 && (stripes & (stripes - 1)) == 0,
             "StripedMutex: stripe count must be a power of two");
  }

  [[nodiscard]] std::mutex& for_key(std::uint64_t key) const noexcept {
    // Fibonacci hashing spreads consecutive ids (neighbours are often inserted close together).
    const std::uint64_t mixed = key * 0x9E3779B97F4A7C15ULL;
    return mutexes_[static_cast<std::size_t>(mixed >> 32U) & mask_];
  }

 private:
  std::size_t mask_;
  std::unique_ptr<std::mutex[]> mutexes_;  // NOLINT(modernize-avoid-c-arrays)
};

}  // namespace vf::detail
