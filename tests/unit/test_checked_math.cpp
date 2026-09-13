#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "core/checked_math.hpp"

namespace {

using vf::detail::checked_add;
using vf::detail::checked_mul;

constexpr auto kMax64 = std::numeric_limits<std::uint64_t>::max();
constexpr auto kMax32 = std::numeric_limits<std::uint32_t>::max();

// Compile-time checks.
static_assert(checked_add(std::uint32_t{1}, std::uint32_t{2}).value() == 3U);
static_assert(!checked_add(kMax32, std::uint32_t{1}).has_value());
static_assert(checked_mul(std::uint32_t{0}, kMax32).value() == 0U);
static_assert(!checked_mul(std::uint32_t{65536}, std::uint32_t{65536}).has_value());

TEST(CheckedMath, AddBoundaries) {
  EXPECT_EQ(checked_add(kMax64 - 1, std::uint64_t{1}).value(), kMax64);
  EXPECT_FALSE(checked_add(kMax64, std::uint64_t{1}).has_value());
  EXPECT_FALSE(checked_add(std::uint64_t{1}, kMax64).has_value());
  EXPECT_EQ(checked_add(std::uint64_t{0}, std::uint64_t{0}).value(), 0U);
}

TEST(CheckedMath, MulBoundaries) {
  EXPECT_EQ(checked_mul(kMax64, std::uint64_t{1}).value(), kMax64);
  EXPECT_EQ(checked_mul(std::uint64_t{0}, kMax64).value(), 0U);
  EXPECT_EQ(checked_mul(kMax64, std::uint64_t{0}).value(), 0U);
  EXPECT_FALSE(checked_mul(kMax64, std::uint64_t{2}).has_value());
  EXPECT_EQ(checked_mul(std::uint64_t{1} << 32U, (std::uint64_t{1} << 32U) - 1).value(),
            0xFFFFFFFF00000000ULL);
  EXPECT_FALSE(checked_mul(std::uint64_t{1} << 32U, std::uint64_t{1} << 32U).has_value());
}

TEST(CheckedMath, TripleMul) {
  // rows * dim * sizeof(float), the typical storage size computation.
  EXPECT_EQ(checked_mul(std::uint64_t{1000000}, std::uint64_t{1536}, std::uint64_t{4}).value(),
            6144000000ULL);
  EXPECT_FALSE(checked_mul(kMax64 / 2, std::uint64_t{1}, std::uint64_t{3}).has_value());
  EXPECT_FALSE(checked_mul(std::uint64_t{3}, kMax64 / 2, std::uint64_t{1}).has_value());
  EXPECT_EQ(checked_mul(std::size_t{0}, std::size_t{7}, std::size_t{9}).value(), 0U);
}

}  // namespace
