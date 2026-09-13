#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <random>

#include "core/rng.hpp"

namespace {

using vf::detail::SplitMix64;
using vf::detail::Xoshiro256ss;

static_assert(std::uniform_random_bit_generator<SplitMix64>);
static_assert(std::uniform_random_bit_generator<Xoshiro256ss>);

// Golden values generated with an independent Python implementation of the reference algorithms
// (https://prng.di.unimi.it/splitmix64.c and xoshiro256starstar.c). The SplitMix64 sequence for
// seed 1234567 also matches the widely published reference output.
TEST(Rng, SplitMix64GoldenSequence) {
  SplitMix64 rng(1234567);
  constexpr std::array<std::uint64_t, 5> kExpected = {
      6457827717110365317ULL, 3203168211198807973ULL, 9817491932198370423ULL,
      4593380528125082431ULL, 16408922859458223821ULL};
  for (const std::uint64_t expected : kExpected) {
    EXPECT_EQ(rng(), expected);
  }
  EXPECT_EQ(vf::detail::splitmix64(0), 16294208416658607535ULL);
  EXPECT_EQ(vf::detail::splitmix64(1234567), kExpected[0]);
}

TEST(Rng, Xoshiro256ssGoldenSequence) {
  Xoshiro256ss rng(42);
  constexpr std::array<std::uint64_t, 5> kExpected = {
      1546998764402558742ULL, 6990951692964543102ULL, 12544586762248559009ULL,
      17057574109182124193ULL, 18295552978065317476ULL};
  for (const std::uint64_t expected : kExpected) {
    EXPECT_EQ(rng(), expected);
  }
}

TEST(Rng, ConstexprEvaluation) {
  constexpr std::uint64_t kFirst = [] {
    Xoshiro256ss rng(42);
    return rng();
  }();
  EXPECT_EQ(kFirst, 1546998764402558742ULL);
}

TEST(Rng, SeedsProduceDistinctStreams) {
  Xoshiro256ss a(1);
  Xoshiro256ss b(2);
  int equal = 0;
  for (int i = 0; i < 1000; ++i) {
    equal += (a() == b()) ? 1 : 0;
  }
  EXPECT_EQ(equal, 0);
}

TEST(Rng, UnitIntervalBounds) {
  const auto max_bits = std::numeric_limits<std::uint64_t>::max();
  EXPECT_EQ(vf::detail::to_unit_interval(0), 0.0);
  EXPECT_LT(vf::detail::to_unit_interval(max_bits), 1.0);
  EXPECT_EQ(vf::detail::to_unit_interval(max_bits), 1.0 - 0x1.0p-53);

  EXPECT_GT(vf::detail::to_unit_interval_open_zero(0), 0.0);
  EXPECT_EQ(vf::detail::to_unit_interval_open_zero(0), 0x1.0p-53);
  EXPECT_EQ(vf::detail::to_unit_interval_open_zero(max_bits), 1.0);
  EXPECT_TRUE(std::isfinite(std::log(vf::detail::to_unit_interval_open_zero(0))));
}

TEST(Rng, UniformFloatStaysInHalfOpenRange) {
  const auto max_bits = std::numeric_limits<std::uint64_t>::max();
  EXPECT_EQ(vf::detail::uniform_float(0, -1.0F, 1.0F), -1.0F);
  EXPECT_LT(vf::detail::uniform_float(max_bits, -1.0F, 1.0F), 1.0F);
  // Narrow range where double->float rounding would otherwise hit `hi`.
  EXPECT_LT(vf::detail::uniform_float(max_bits, 1.0F, std::nextafter(1.0F, 2.0F)),
            std::nextafter(1.0F, 2.0F));

  Xoshiro256ss rng(7);
  for (int i = 0; i < 100000; ++i) {
    const float x = vf::detail::uniform_float(rng(), 2.0F, 3.0F);
    ASSERT_GE(x, 2.0F);
    ASSERT_LT(x, 3.0F);
  }
}

TEST(Rng, MulHighMatchesReference) {
  constexpr auto kMax = std::numeric_limits<std::uint64_t>::max();
  EXPECT_EQ(vf::detail::mul_high_u64(0, kMax), 0U);
  EXPECT_EQ(vf::detail::mul_high_u64(1, kMax), 0U);
  EXPECT_EQ(vf::detail::mul_high_u64(kMax, kMax), kMax - 1);  // (2^64-1)^2 = 2^128 - 2^65 + 1
  EXPECT_EQ(vf::detail::mul_high_u64(1ULL << 32U, 1ULL << 32U), 1U);
  EXPECT_EQ(vf::detail::mul_high_u64(0xFFFFFFFFULL, 0xFFFFFFFFULL), 0U);
  // 0x123456789ABCDEF0 * 0x0FEDCBA987654321 = 0x0121FA00AD77D742_2236D88FE5618CF0 (checked in
  // Python).
  EXPECT_EQ(vf::detail::mul_high_u64(0x123456789ABCDEF0ULL, 0x0FEDCBA987654321ULL),
            0x0121FA00AD77D742ULL);
}

TEST(Rng, UniformBelowRangeAndBalance) {
  Xoshiro256ss rng(123);
  EXPECT_EQ(vf::detail::uniform_below(rng, 1), 0U);
  std::array<int, 7> histogram{};
  constexpr int kDraws = 70000;
  for (int i = 0; i < kDraws; ++i) {
    const std::uint64_t v = vf::detail::uniform_below(rng, 7);
    ASSERT_LT(v, 7U);
    ++histogram[v];
  }
  for (const int count : histogram) {
    EXPECT_NEAR(count, kDraws / 7, 400);  // ~4.4 standard deviations
  }
  // Huge bounds exercise the rejection threshold computation.
  const std::uint64_t big = (std::uint64_t{1} << 63U) + 12345;
  for (int i = 0; i < 1000; ++i) {
    ASSERT_LT(vf::detail::uniform_below(rng, big), big);
  }
}

TEST(Rng, UniformMeanIsPlausible) {
  // Loose statistical sanity check; the stream is deterministic, so this never flakes.
  Xoshiro256ss rng(99);
  constexpr int kSamples = 200000;
  double sum = 0.0;
  for (int i = 0; i < kSamples; ++i) {
    sum += vf::detail::to_unit_interval(rng());
  }
  EXPECT_NEAR(sum / kSamples, 0.5, 0.005);
}

}  // namespace
