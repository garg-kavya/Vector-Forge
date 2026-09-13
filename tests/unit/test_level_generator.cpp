#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/rng.hpp"
#include "index/hnsw/level_generator.hpp"

namespace {

using vf::detail::LevelGenerator;

// Paper formula evaluated in long double; used only away from level boundaries.
int formula_level(std::uint64_t seed, std::uint32_t id, std::uint32_t m, int cap) {
  const std::uint64_t bits = vf::detail::splitmix64(seed ^ vf::detail::splitmix64(id));
  const long double u = static_cast<long double>((bits >> 11U) + 1U) * 0x1.0p-53L;
  const long double x = -std::log(u) / std::log(static_cast<long double>(m));
  const long double nearest = std::round(x);
  if (std::fabs(x - nearest) < 1e-9L) {
    return -1;  // too close to an integer boundary to compare reliably in floating point
  }
  const int level = static_cast<int>(std::floor(x));
  return level < cap ? level : cap;
}

TEST(LevelGenerator, DeterministicAndIdDerived) {
  const LevelGenerator a(123, 16, 16);
  const LevelGenerator b(123, 16, 16);
  const LevelGenerator other_seed(124, 16, 16);
  std::size_t differences = 0;
  for (std::uint32_t id = 0; id < 20000; ++id) {
    ASSERT_EQ(a.level_for(id), b.level_for(id));
    ASSERT_EQ(a.level_for(id), a.level_for(id));
    differences += a.level_for(id) != other_seed.level_for(id) ? 1U : 0U;
  }
  EXPECT_GT(differences, 0U) << "the seed must influence levels";
}

TEST(LevelGenerator, MatchesPaperFormula) {
  for (const std::uint32_t m : {2U, 5U, 16U, 48U}) {
    const LevelGenerator gen(0xABCDEF, m, 16);
    std::size_t compared = 0;
    for (std::uint32_t id = 0; id < 100000; ++id) {
      const int expected = formula_level(0xABCDEF, id, m, 16);
      if (expected < 0) {
        continue;
      }
      ++compared;
      ASSERT_EQ(gen.level_for(id), expected) << "m=" << m << " id=" << id;
    }
    EXPECT_GT(compared, 99000U);
  }
}

TEST(LevelGenerator, GoldenValues) {
  // Pins the exact integer algorithm (any change alters every persisted graph). Expected values
  // were computed with an independent arbitrary-precision Python implementation of §9.2.
  const LevelGenerator gen(0x5EEDF0A6EULL, 16, 16);
  auto first_id_with_level = [&](std::uint8_t level) {
    for (std::uint32_t id = 0;; ++id) {
      if (gen.level_for(id) >= level) {
        return id;
      }
    }
  };
  EXPECT_EQ(first_id_with_level(1), 37U);
  EXPECT_EQ(first_id_with_level(2), 38U);
  EXPECT_EQ(first_id_with_level(3), 3149U);
  EXPECT_EQ(first_id_with_level(4), 357662U);
  std::uint64_t level_sum = 0;
  for (std::uint32_t id = 0; id < 100000; ++id) {
    level_sum += gen.level_for(id);
  }
  EXPECT_EQ(level_sum, 6588U);
}

TEST(LevelGenerator, RespectsCap) {
  // M = 2 gives high levels often; with cap 3 nothing may exceed it.
  const LevelGenerator gen(7, 2, 3);
  std::size_t at_cap = 0;
  for (std::uint32_t id = 0; id < 10000; ++id) {
    ASSERT_LE(gen.level_for(id), 3U);
    at_cap += gen.level_for(id) == 3 ? 1U : 0U;
  }
  EXPECT_GT(at_cap, 0U);
}

TEST(LevelGenerator, DistributionMatchesGeometricLaw) {
  // P(level >= l) = M^-l. Over 10^6 ids each tail count is binomial; allow 6 standard deviations.
  constexpr std::uint32_t kIds = 1000000;
  for (const std::uint32_t m : {4U, 16U}) {
    const LevelGenerator gen(99, m, 16);
    std::vector<std::uint64_t> at_least(17, 0);
    for (std::uint32_t id = 0; id < kIds; ++id) {
      const std::uint8_t level = gen.level_for(id);
      for (std::uint8_t l = 0; l <= level; ++l) {
        ++at_least[l];
      }
    }
    EXPECT_EQ(at_least[0], kIds);
    for (std::size_t l = 1; l < at_least.size(); ++l) {
      const double p = std::pow(static_cast<double>(m), -static_cast<double>(l));
      const double mean = p * kIds;
      if (mean < 20.0) {
        break;
      }
      const double sd = std::sqrt(mean * (1.0 - p));
      EXPECT_NEAR(static_cast<double>(at_least[l]), mean, 6.0 * sd) << "m=" << m << " level " << l;
    }
  }
}

}  // namespace
