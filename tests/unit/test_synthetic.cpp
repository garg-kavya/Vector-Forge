#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include <vectorforge/types.hpp>

#include "util/synthetic.hpp"

namespace {

using vf::detail::SyntheticDistribution;
using vf::detail::SyntheticGenerator;
using vf::detail::SyntheticSpec;

std::vector<float> generate(const SyntheticSpec& spec, std::size_t rows) {
  SyntheticGenerator gen = SyntheticGenerator::create(spec).value();
  std::vector<float> out(rows * spec.dim);
  for (std::size_t r = 0; r < rows; ++r) {
    gen.next_row(std::span<float>(out).subspan(r * spec.dim, spec.dim));
  }
  return out;
}

TEST(Synthetic, Validation) {
  SyntheticSpec s;
  EXPECT_EQ(SyntheticGenerator::create(s).status().code(),
            vf::ErrorCode::InvalidArgument);  // dim 0
  s.dim = 4;
  EXPECT_TRUE(SyntheticGenerator::create(s).ok());
  s.clusters = 0;
  EXPECT_FALSE(SyntheticGenerator::create(s).ok());
  s.clusters = 10;
  s.spread = -1.0F;
  EXPECT_FALSE(SyntheticGenerator::create(s).ok());
  s.spread = 0.1F;
  s.dim = 65536;
  s.clusters = 1U << 13U;  // 2^29 centre components
  EXPECT_FALSE(SyntheticGenerator::create(s).ok());
  s.distribution = SyntheticDistribution::Uniform;
  EXPECT_TRUE(SyntheticGenerator::create(s).ok()) << "cluster limits only apply to mixtures";
}

TEST(Synthetic, DeterministicPerSeed) {
  for (const auto dist : {SyntheticDistribution::Uniform, SyntheticDistribution::GaussianMixture}) {
    SyntheticSpec s;
    s.distribution = dist;
    s.dim = 8;
    s.seed = 5;
    const auto a = generate(s, 50);
    const auto b = generate(s, 50);
    EXPECT_EQ(a, b);
    s.seed = 6;
    EXPECT_NE(generate(s, 50), a);
  }
}

TEST(Synthetic, UniformGoldenValuesAndRange) {
  SyntheticSpec s;
  s.distribution = SyntheticDistribution::Uniform;
  s.dim = 3;
  s.seed = 42;
  const auto v = generate(s, 2000);
  for (float x : v) {
    ASSERT_GE(x, -1.0F);
    ASSERT_LT(x, 1.0F);
  }
  // Uniform output is fully specified (xoshiro256** + 53-bit conversion): identical on all
  // platforms. First component = -1 + 2 * ((1546998764402558742 >> 11) * 2^-53).
  const double u = static_cast<double>(1546998764402558742ULL >> 11U) * 0x1.0p-53;
  EXPECT_EQ(v[0], static_cast<float>(-1.0 + (2.0 * u)));
}

TEST(Synthetic, MixtureRowsClusterAroundSharedCentres) {
  SyntheticSpec base;
  base.dim = 32;
  base.clusters = 4;
  base.spread = 0.01F;
  base.center_range = 1.0F;
  base.seed = 1;
  base.mixture_seed = 77;
  SyntheticSpec queries = base;
  queries.seed = 2;  // different samples, same mixture

  const auto b = generate(base, 400);
  const auto q = generate(queries, 50);
  // With a tiny spread every query row lies very close to some base row (same centre).
  for (std::size_t i = 0; i < 50; ++i) {
    double best = 1e300;
    for (std::size_t j = 0; j < 400; ++j) {
      double d = 0.0;
      for (std::size_t t = 0; t < 32; ++t) {
        const double diff =
            static_cast<double>(q[(i * 32) + t]) - static_cast<double>(b[(j * 32) + t]);
        d += diff * diff;
      }
      best = std::min(best, d);
    }
    // Two draws from one centre differ by ~sqrt(2*32)*0.01 = 0.08 in L2 (0.0064 squared);
    // distinct centres are ~sqrt(32*2/3) = 4.6 apart.
    EXPECT_LT(best, 0.1) << "query " << i;
  }
}

TEST(Synthetic, GaussianMomentsArePlausible) {
  SyntheticSpec s;
  s.dim = 1;
  s.clusters = 1;
  s.center_range = 0.0F;  // centre at 0
  s.spread = 1.0F;
  const auto v = generate(s, 200000);
  double mean = 0.0;
  double var = 0.0;
  for (float x : v) {
    mean += static_cast<double>(x);
  }
  mean /= static_cast<double>(v.size());
  for (float x : v) {
    const double dx = static_cast<double>(x) - mean;
    var += dx * dx;
  }
  var /= static_cast<double>(v.size());
  EXPECT_NEAR(mean, 0.0, 0.01);
  EXPECT_NEAR(var, 1.0, 0.02);
}

}  // namespace
