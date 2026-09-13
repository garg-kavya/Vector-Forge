// Distance kernel correctness.
//
// Oracle: double-precision naive sums. Tolerance: a rigorous first-order bound for summing n
// float products in any order: |computed - exact| <= ((1 + eps)^(n + 4) - 1) * sum|term_i|,
// covering product/difference rounding and n - 1 rounded additions. Tests therefore never depend
// on lucky rounding and do not flake.

#include <gtest/gtest.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include <vectorforge/simd.hpp>

#include "core/aligned_alloc.hpp"
#include "core/rng.hpp"
#include "simd/kernels.hpp"
#include "simd/metric_distance.hpp"
#include "support/test_data.hpp"

namespace {

using vf::detail::KernelTable;

struct Reference {
  double value = 0.0;
  double abs_terms = 0.0;
};

Reference ref_dot(const float* a, const float* b, std::size_t d) {
  Reference r;
  for (std::size_t i = 0; i < d; ++i) {
    const double t = static_cast<double>(a[i]) * static_cast<double>(b[i]);
    r.value += t;
    r.abs_terms += std::fabs(t);
  }
  return r;
}

Reference ref_l2sq(const float* a, const float* b, std::size_t d) {
  Reference r;
  for (std::size_t i = 0; i < d; ++i) {
    const double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    r.value += diff * diff;
    r.abs_terms += diff * diff;
  }
  return r;
}

double error_bound(std::size_t d, double abs_terms) {
  const double eps = static_cast<double>(FLT_EPSILON);
  return (std::pow(1.0 + eps, static_cast<double>(d) + 4.0) - 1.0) * abs_terms + 1e-30;
}

void expect_close(float actual, const Reference& ref, std::size_t d, const std::string& what) {
  const double bound = error_bound(d, ref.abs_terms);
  EXPECT_LE(std::fabs(static_cast<double>(actual) - ref.value), bound)
      << what << " d=" << d << " actual=" << actual << " ref=" << ref.value << " bound=" << bound;
}

std::vector<std::size_t> test_dims() {
  std::vector<std::size_t> dims;
  for (std::size_t d = 0; d <= 67; ++d) {
    dims.push_back(d);
  }
  for (std::size_t d : {100U, 128U, 384U, 768U, 1536U, 1537U, 1543U}) {
    dims.push_back(d);
  }
  return dims;
}

const KernelTable* const kTables[] = {&vf::detail::scalar_kernel_table(),
                                      &vf::detail::scalar_autovec_kernel_table()};

TEST(Kernels, ActiveTableIsScalarInPhase1) {
  const KernelTable& k = vf::detail::kernels();
  EXPECT_EQ(k.level, vf::SimdLevel::Scalar);
  EXPECT_EQ(k.name, "scalar");
  EXPECT_EQ(vf::active_simd_level(), vf::SimdLevel::Scalar);
  EXPECT_EQ(vf::to_string(vf::SimdLevel::Scalar), "scalar");
  EXPECT_EQ(vf::to_string(vf::SimdLevel::Avx2), "avx2");
  for (const KernelTable* t : kTables) {
    EXPECT_NE(t->dot, nullptr);
    EXPECT_NE(t->l2sq, nullptr);
    EXPECT_NE(t->norm2, nullptr);
    EXPECT_NE(t->dot_1_to_n, nullptr);
    EXPECT_NE(t->l2sq_1_to_n, nullptr);
  }
}

TEST(Kernels, MatchDoubleReferenceAcrossDimsAndOffsets) {
  vf::detail::Xoshiro256ss rng(2024);
  for (std::size_t d : test_dims()) {
    // Offsets in whole floats (0..7) from a 64-byte-aligned base: byte offsets 0..28 exercise every
    // misalignment relative to 8/16/32-byte SIMD boundaries without violating float alignment.
    const auto a_buf = vf::detail::make_aligned_array<float>(d + 8);
    const auto b_buf = vf::detail::make_aligned_array<float>(d + 8);
    const std::vector<float> a_src = vf::test::random_vector(rng, d, -2.0F, 2.0F);
    const std::vector<float> b_src = vf::test::random_vector(rng, d, -2.0F, 2.0F);
    for (std::size_t offset = 0; offset < 8; ++offset) {
      float* a = a_buf.get() + offset;
      float* b = b_buf.get() + offset;
      std::copy(a_src.begin(), a_src.end(), a);
      std::copy(b_src.begin(), b_src.end(), b);
      const Reference rd = ref_dot(a, b, d);
      const Reference rl = ref_l2sq(a, b, d);
      const Reference rn = ref_dot(a, a, d);
      for (const KernelTable* t : kTables) {
        const std::string name(t->name);
        expect_close(t->dot(a, b, d), rd, d, name + ".dot");
        expect_close(t->l2sq(a, b, d), rl, d, name + ".l2sq");
        expect_close(t->norm2(a, d), rn, d, name + ".norm2");
      }
    }
  }
}

TEST(Kernels, ScalarIsBitIdenticalAcrossOffsets) {
  const KernelTable& t = vf::detail::scalar_kernel_table();
  vf::detail::Xoshiro256ss rng(5);
  for (std::size_t d : {1U, 7U, 31U, 128U, 1537U}) {
    const std::vector<float> a_src = vf::test::random_vector(rng, d);
    const std::vector<float> b_src = vf::test::random_vector(rng, d);
    const auto a_buf = vf::detail::make_aligned_array<float>(d + 8);
    const auto b_buf = vf::detail::make_aligned_array<float>(d + 8);
    float dot0 = 0.0F;
    float l2_0 = 0.0F;
    for (std::size_t offset = 0; offset < 8; ++offset) {
      std::copy(a_src.begin(), a_src.end(), a_buf.get() + offset);
      std::copy(b_src.begin(), b_src.end(), b_buf.get() + offset);
      const float dot = t.dot(a_buf.get() + offset, b_buf.get() + offset, d);
      const float l2 = t.l2sq(a_buf.get() + offset, b_buf.get() + offset, d);
      if (offset == 0) {
        dot0 = dot;
        l2_0 = l2;
      } else {
        EXPECT_EQ(dot, dot0) << "d=" << d << " offset=" << offset;
        EXPECT_EQ(l2, l2_0) << "d=" << d << " offset=" << offset;
      }
    }
  }
}

TEST(Kernels, BatchMatchesPairwise) {
  vf::detail::Xoshiro256ss rng(77);
  constexpr std::size_t kRows = 37;
  for (std::size_t d : {1U, 9U, 128U, 385U}) {
    const std::vector<float> q = vf::test::random_vector(rng, d);
    const std::vector<float> rows = vf::test::random_matrix(rng, kRows, d);
    std::vector<float> out_dot(kRows);
    std::vector<float> out_l2(kRows);
    for (const KernelTable* t : kTables) {
      t->dot_1_to_n(q.data(), rows.data(), kRows, d, out_dot.data());
      t->l2sq_1_to_n(q.data(), rows.data(), kRows, d, out_l2.data());
      for (std::size_t r = 0; r < kRows; ++r) {
        const float* row = rows.data() + (r * d);
        if (t == &vf::detail::scalar_kernel_table()) {
          // Same strict evaluation order: exact equality.
          EXPECT_EQ(out_dot[r], t->dot(q.data(), row, d));
          EXPECT_EQ(out_l2[r], t->l2sq(q.data(), row, d));
        } else {
          expect_close(out_dot[r], ref_dot(q.data(), row, d), d, "autovec.dot_1_to_n");
          expect_close(out_l2[r], ref_l2sq(q.data(), row, d), d, "autovec.l2sq_1_to_n");
        }
      }
    }
  }
}

TEST(Kernels, SpecialValues) {
  for (const KernelTable* t : kTables) {
    std::vector<float> zeros(1536, 0.0F);
    EXPECT_EQ(t->dot(zeros.data(), zeros.data(), zeros.size()), 0.0F);
    EXPECT_EQ(t->l2sq(zeros.data(), zeros.data(), zeros.size()), 0.0F);
    EXPECT_EQ(t->norm2(zeros.data(), zeros.size()), 0.0F);

    // Identical vectors: every difference is exactly zero.
    vf::detail::Xoshiro256ss rng(3);
    const std::vector<float> v = vf::test::random_vector(rng, 777, -1e6F, 1e6F);
    EXPECT_EQ(t->l2sq(v.data(), v.data(), v.size()), 0.0F);

    // Largest accepted magnitude, opposite signs, maximal dimension: must stay finite.
    std::vector<float> big(vf::kMaxDim, vf::kMaxAbsComponent);
    std::vector<float> neg(vf::kMaxDim, -vf::kMaxAbsComponent);
    EXPECT_TRUE(std::isfinite(t->l2sq(big.data(), neg.data(), big.size())));
    EXPECT_TRUE(std::isfinite(t->dot(big.data(), neg.data(), big.size())));
    EXPECT_TRUE(std::isfinite(t->norm2(big.data(), big.size())));

    // Subnormal inputs underflow gracefully (no NaN).
    std::vector<float> tiny(64, FLT_TRUE_MIN);
    const float n = t->norm2(tiny.data(), tiny.size());
    EXPECT_FALSE(std::isnan(n));
    EXPECT_GE(n, 0.0F);

    // Empty input.
    EXPECT_EQ(t->dot(nullptr, nullptr, 0), 0.0F);
    EXPECT_EQ(t->l2sq(nullptr, nullptr, 0), 0.0F);
  }
}

TEST(Kernels, KnownValues) {
  const std::vector<float> a = {1.0F, 2.0F, 3.0F};
  const std::vector<float> b = {4.0F, -5.0F, 6.0F};
  for (const KernelTable* t : kTables) {
    EXPECT_EQ(t->dot(a.data(), b.data(), 3), 12.0F);   // 4 - 10 + 18
    EXPECT_EQ(t->l2sq(a.data(), b.data(), 3), 67.0F);  // 9 + 49 + 9
    EXPECT_EQ(t->norm2(b.data(), 3), 77.0F);           // 16 + 25 + 36
  }
}

TEST(Kernels, MetricDistanceConvention) {
  const KernelTable& t = vf::detail::kernels();
  const std::vector<float> a = {0.6F, 0.8F};
  const std::vector<float> b = {0.8F, 0.6F};
  const float dot = t.dot(a.data(), b.data(), 2);
  EXPECT_EQ(vf::detail::metric_distance(t, vf::Metric::L2, a.data(), b.data(), 2),
            t.l2sq(a.data(), b.data(), 2));
  EXPECT_EQ(vf::detail::metric_distance(t, vf::Metric::InnerProduct, a.data(), b.data(), 2), -dot);
  EXPECT_EQ(vf::detail::metric_distance(t, vf::Metric::Cosine, a.data(), b.data(), 2), 1.0F - dot);
}

}  // namespace
