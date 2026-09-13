#include <gtest/gtest.h>

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <vectorforge/distance.hpp>

#include "core/rng.hpp"
#include "core/vector_ops.hpp"
#include "simd/kernels.hpp"
#include "support/test_data.hpp"

namespace {

using vf::ErrorCode;
using vf::Metric;

double norm_double(const std::vector<float>& v) {
  double s = 0.0;
  for (float x : v) {
    s += static_cast<double>(x) * static_cast<double>(x);
  }
  return std::sqrt(s);
}

TEST(Normalize, ProducesUnitNormAndPreservesDirection) {
  vf::detail::Xoshiro256ss rng(11);
  for (std::size_t d : {1U, 2U, 3U, 17U, 128U, 1536U}) {
    const std::vector<float> original = vf::test::random_vector(rng, d, -10.0F, 10.0F);
    std::vector<float> v = original;
    ASSERT_TRUE(vf::normalize(v).ok());
    EXPECT_NEAR(norm_double(v), 1.0, 1e-5) << "d=" << d;
    const double scale = 1.0 / norm_double(original);
    for (std::size_t i = 0; i < d; ++i) {
      EXPECT_NEAR(static_cast<double>(v[i]), static_cast<double>(original[i]) * scale, 1e-6);
    }
  }
}

TEST(Normalize, AlreadyUnitVectorIsStable) {
  std::vector<float> v = {0.6F, 0.8F};
  ASSERT_TRUE(vf::normalize(v).ok());
  EXPECT_NEAR(v[0], 0.6F, 1e-7F);
  EXPECT_NEAR(v[1], 0.8F, 1e-7F);
}

TEST(Normalize, LargeMagnitudes) {
  std::vector<float> v(1536, vf::kMaxAbsComponent);
  ASSERT_TRUE(vf::normalize(v).ok());
  EXPECT_NEAR(norm_double(v), 1.0, 1e-5);
}

TEST(Normalize, RejectsZeroEmptyAndTinyVectors) {
  std::vector<float> zeros(8, 0.0F);
  const std::vector<float> zeros_copy = zeros;
  EXPECT_EQ(vf::normalize(zeros).code(), ErrorCode::InvalidArgument);
  EXPECT_EQ(zeros, zeros_copy);  // unchanged on error

  std::vector<float> empty;
  EXPECT_EQ(vf::normalize(empty).code(), ErrorCode::InvalidArgument);

  std::vector<float> tiny(4, FLT_TRUE_MIN);  // norm^2 underflows to zero
  const std::vector<float> tiny_copy = tiny;
  EXPECT_EQ(vf::normalize(tiny).code(), ErrorCode::InvalidArgument);
  EXPECT_EQ(tiny, tiny_copy);
}

TEST(Normalize, RejectsNonFinite) {
  std::vector<float> v = {1.0F, std::numeric_limits<float>::quiet_NaN()};
  EXPECT_EQ(vf::normalize(v).code(), ErrorCode::InvalidArgument);
  v = {1.0F, std::numeric_limits<float>::infinity()};
  EXPECT_EQ(vf::normalize(v).code(), ErrorCode::InvalidArgument);
  v = {1.0F, 2e16F};
  EXPECT_EQ(vf::normalize(v).code(), ErrorCode::InvalidArgument);
}

TEST(Normalize, InternalWorksWithEveryTable) {
  for (const auto* t :
       {&vf::detail::scalar_kernel_table(), &vf::detail::scalar_autovec_kernel_table()}) {
    std::vector<float> v = {3.0F, 4.0F};
    ASSERT_TRUE(vf::detail::normalize_inplace(v, *t).ok());
    EXPECT_FLOAT_EQ(v[0], 0.6F);
    EXPECT_FLOAT_EQ(v[1], 0.8F);
  }
}

TEST(Distance, KnownValues) {
  const std::vector<float> a = {1.0F, 2.0F};
  const std::vector<float> b = {4.0F, 6.0F};
  EXPECT_EQ(vf::distance(Metric::L2, a, b).value(), 25.0F);
  EXPECT_EQ(vf::distance(Metric::InnerProduct, a, b).value(), -16.0F);

  const std::vector<float> x = {1.0F, 0.0F};
  const std::vector<float> y = {0.0F, 5.0F};
  const std::vector<float> x_scaled = {7.0F, 0.0F};
  const std::vector<float> x_neg = {-3.0F, 0.0F};
  EXPECT_NEAR(vf::distance(Metric::Cosine, x, x_scaled).value(), 0.0F, 1e-7F);
  EXPECT_NEAR(vf::distance(Metric::Cosine, x, y).value(), 1.0F, 1e-7F);
  EXPECT_NEAR(vf::distance(Metric::Cosine, x, x_neg).value(), 2.0F, 1e-7F);
}

TEST(Distance, LowerIsBetterOrdering) {
  const std::vector<float> q = {1.0F, 0.0F};
  const std::vector<float> near = {0.9F, 0.1F};
  const std::vector<float> far = {-1.0F, 0.2F};
  for (const Metric m : {Metric::L2, Metric::InnerProduct, Metric::Cosine}) {
    EXPECT_LT(vf::distance(m, q, near).value(), vf::distance(m, q, far).value())
        << vf::to_string(m);
  }
}

TEST(Distance, Errors) {
  const std::vector<float> a = {1.0F, 2.0F};
  const std::vector<float> b = {1.0F, 2.0F, 3.0F};
  const std::vector<float> empty;
  const std::vector<float> zero = {0.0F, 0.0F};
  const std::vector<float> nan = {std::numeric_limits<float>::quiet_NaN(), 1.0F};
  EXPECT_EQ(vf::distance(Metric::L2, a, b).status().code(), ErrorCode::DimensionMismatch);
  EXPECT_EQ(vf::distance(Metric::L2, empty, empty).status().code(), ErrorCode::InvalidArgument);
  EXPECT_EQ(vf::distance(Metric::Cosine, a, zero).status().code(), ErrorCode::InvalidArgument);
  EXPECT_EQ(vf::distance(Metric::L2, a, nan).status().code(), ErrorCode::InvalidArgument);
  EXPECT_EQ(vf::distance(static_cast<Metric>(9), a, a).status().code(), ErrorCode::InvalidArgument);
  // L2 and IP are well defined for zero vectors.
  EXPECT_EQ(vf::distance(Metric::L2, zero, zero).value(), 0.0F);
  EXPECT_EQ(vf::distance(Metric::InnerProduct, zero, a).value(), 0.0F);
}

}  // namespace
