#include <gtest/gtest.h>

#include <limits>
#include <vector>

#include <vectorforge/types.hpp>

#include "util/recall.hpp"

namespace {

using vf::ErrorCode;
using vf::kInvalidExternalId;
using vf::detail::recall_at_k;

constexpr float kInf = std::numeric_limits<float>::infinity();

TEST(Recall, PerfectAndPartial) {
  // Two queries, k = 3.
  const std::vector<vf::ExternalId> exact = {1, 2, 3, 10, 20, 30};
  const std::vector<float> exact_d = {0.1F, 0.2F, 0.3F, 1.0F, 2.0F, 3.0F};
  const std::vector<vf::ExternalId> approx = {3, 2, 1, 10, 99, 98};
  const std::vector<float> approx_d = {0.3F, 0.2F, 0.1F, 1.0F, 5.0F, 6.0F};
  const auto r = recall_at_k(approx, approx_d, 3, exact, exact_d, 3, 2, 3);
  ASSERT_TRUE(r.ok());
  EXPECT_DOUBLE_EQ(r.value().mean, (1.0 + (1.0 / 3.0)) / 2.0);
  EXPECT_DOUBLE_EQ(r.value().min, 1.0 / 3.0);
  EXPECT_EQ(r.value().queries, 2U);
}

TEST(Recall, TiesCountAsHits) {
  // Exact top-2 is {1, 2} at distance 1.0; id 7 is equally distant and must count.
  const std::vector<vf::ExternalId> exact = {1, 2};
  const std::vector<float> exact_d = {0.5F, 1.0F};
  const std::vector<vf::ExternalId> approx = {1, 7};
  const std::vector<float> approx_d = {0.5F, 1.0F};
  EXPECT_DOUBLE_EQ(recall_at_k(approx, approx_d, 2, exact, exact_d, 2, 1, 2).value().mean, 1.0);
  // Clearly worse alternative does not count.
  const std::vector<float> worse_d = {0.5F, 1.01F};
  EXPECT_DOUBLE_EQ(recall_at_k(approx, worse_d, 2, exact, exact_d, 2, 1, 2).value().mean, 0.5);
}

TEST(Recall, UsesFirstKColumnsOnly) {
  const std::vector<vf::ExternalId> exact = {1, 2, 3, 4};  // exact_cols = 4
  const std::vector<float> exact_d = {1, 2, 3, 4};
  const std::vector<vf::ExternalId> approx = {1, 3};  // approx_cols = 2
  const std::vector<float> approx_d = {1, 3};
  // k = 1: only id 1 matters.
  EXPECT_DOUBLE_EQ(recall_at_k(approx, approx_d, 2, exact, exact_d, 4, 1, 1).value().mean, 1.0);
  // k = 2: id 3 is not in the exact top-2 and is farther than distance 2.
  EXPECT_DOUBLE_EQ(recall_at_k(approx, approx_d, 2, exact, exact_d, 4, 1, 2).value().mean, 0.5);
}

TEST(Recall, PaddingAndDuplicates) {
  // Only two real exact neighbours exist.
  const std::vector<vf::ExternalId> exact = {5, 6, kInvalidExternalId};
  const std::vector<float> exact_d = {1.0F, 2.0F, kInf};
  const std::vector<vf::ExternalId> approx = {5, 5, kInvalidExternalId};
  const std::vector<float> approx_d = {1.0F, 1.0F, kInf};
  EXPECT_DOUBLE_EQ(recall_at_k(approx, approx_d, 3, exact, exact_d, 3, 1, 3).value().mean, 0.5)
      << "duplicate counts once; denominator is the 2 valid exact results";

  const std::vector<vf::ExternalId> none = {kInvalidExternalId};
  const std::vector<float> none_d = {kInf};
  EXPECT_DOUBLE_EQ(recall_at_k(none, none_d, 1, none, none_d, 1, 1, 1).value().mean, 1.0)
      << "no exact results -> recall 1";
}

TEST(Recall, NegativeDistancesUseRelativeTolerance) {
  const std::vector<vf::ExternalId> exact = {1};
  const std::vector<float> exact_d = {-100.0F};
  const std::vector<vf::ExternalId> approx = {2};
  const std::vector<float> within = {-99.99995F};  // 5e-5 above: within 1e-6 * 100 = 1e-4
  const std::vector<float> outside = {-99.9F};
  EXPECT_DOUBLE_EQ(recall_at_k(approx, within, 1, exact, exact_d, 1, 1, 1).value().mean, 1.0);
  EXPECT_DOUBLE_EQ(recall_at_k(approx, outside, 1, exact, exact_d, 1, 1, 1).value().mean, 0.0);
}

TEST(Recall, Errors) {
  const std::vector<vf::ExternalId> ids = {1, 2};
  const std::vector<float> d = {1, 2};
  EXPECT_EQ(recall_at_k(ids, d, 2, ids, d, 2, 1, 0).status().code(), ErrorCode::InvalidArgument);
  EXPECT_EQ(recall_at_k(ids, d, 2, ids, d, 2, 1, 3).status().code(), ErrorCode::InvalidArgument);
  EXPECT_EQ(recall_at_k(ids, d, 2, ids, d, 2, 2, 1).status().code(), ErrorCode::InvalidArgument);
  const auto empty = recall_at_k({}, {}, 2, {}, {}, 2, 0, 1);
  ASSERT_TRUE(empty.ok());
  EXPECT_EQ(empty.value().queries, 0U);
}

}  // namespace
