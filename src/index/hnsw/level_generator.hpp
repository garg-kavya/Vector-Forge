#pragma once

// Deterministic HNSW level assignment (docs/DESIGN.md §9.2).
//
// The paper draws level = floor(-ln(U) / ln(M)) with U ~ Uniform(0, 1]. Here U is derived from the
// node id only: bits = splitmix64(seed ^ splitmix64(id)), v = (bits >> 11) + 1 in [1, 2^53] and
// U = v * 2^-53. Because floor(-ln(U) / ln(M)) >= l  <=>  U <= M^-l  <=>  v * M^l <= 2^53, the
// level is computed with exact integer arithmetic: the largest l (capped at max_level) with M^l <=
// floor(2^53 / v). This is the paper's formula evaluated exactly, without std::log, so the
// assignment is bit-identical on every platform and independent of insertion interleaving.

#include <cstdint>

#include <vectorforge/types.hpp>

#include "core/rng.hpp"

namespace vf::detail {

class LevelGenerator {
 public:
  // Preconditions: m >= 2 (HnswParams::validate()).
  constexpr LevelGenerator(std::uint64_t seed, std::uint32_t m, std::uint8_t max_level) noexcept
      : seed_(seed), m_(m), max_level_(max_level) {}

  [[nodiscard]] constexpr std::uint8_t level_for(InternalId id) const noexcept {
    const std::uint64_t bits = splitmix64(seed_ ^ splitmix64(id));
    const std::uint64_t v = (bits >> 11U) + 1U;  // [1, 2^53]
    const std::uint64_t bound = kTwoPow53 / v;   // level >= l  <=>  M^l <= bound
    std::uint64_t power = 1;                     // M^level, <= 2^53, so power * M < 2^62
    std::uint8_t level = 0;
    while (level < max_level_) {
      const std::uint64_t next = power * m_;
      if (next > bound) {
        break;
      }
      power = next;
      ++level;
    }
    return level;
  }

  [[nodiscard]] constexpr std::uint64_t seed() const noexcept { return seed_; }

 private:
  static constexpr std::uint64_t kTwoPow53 = std::uint64_t{1} << 53U;

  std::uint64_t seed_;
  std::uint64_t m_;
  std::uint8_t max_level_;
};

}  // namespace vf::detail
