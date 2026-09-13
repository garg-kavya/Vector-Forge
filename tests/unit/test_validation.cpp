#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <vectorforge/types.hpp>

#include "core/validation.hpp"

namespace {

using vf::ErrorCode;
using vf::detail::find_invalid_component;
using vf::detail::validate_batch;
using vf::detail::validate_vector;

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr float kInf = std::numeric_limits<float>::infinity();

TEST(Validation, FindInvalidComponent) {
  EXPECT_EQ(find_invalid_component(std::vector<float>{}), 0U);
  EXPECT_EQ(find_invalid_component(std::vector<float>{0.0F, -1.0F, 1e16F, -1e16F}), 4U);
  EXPECT_EQ(find_invalid_component(std::vector<float>{1.0F, kNaN}), 1U);
  EXPECT_EQ(find_invalid_component(std::vector<float>{-kInf, 1.0F}), 0U);
  EXPECT_EQ(find_invalid_component(std::vector<float>{1.0F, 2.0F, std::nextafter(1e16F, 1e17F)}),
            2U);
  EXPECT_EQ(
      find_invalid_component(std::vector<float>{-0.0F, std::numeric_limits<float>::denorm_min()}),
      2U);
}

TEST(Validation, ValidateVector) {
  EXPECT_TRUE(validate_vector(std::vector<float>{1.0F, 2.0F, 3.0F}, 3).ok());

  const vf::Status wrong_dim = validate_vector(std::vector<float>{1.0F, 2.0F}, 3);
  EXPECT_EQ(wrong_dim.code(), ErrorCode::DimensionMismatch);
  EXPECT_NE(wrong_dim.message().find("expected dimension 3, got 2"), std::string::npos);

  const vf::Status nan = validate_vector(std::vector<float>{1.0F, kNaN, 3.0F}, 3);
  EXPECT_EQ(nan.code(), ErrorCode::InvalidArgument);
  EXPECT_NE(nan.message().find("component 1 is NaN"), std::string::npos);

  const vf::Status inf = validate_vector(std::vector<float>{kInf, 1.0F, 3.0F}, 3);
  EXPECT_EQ(inf.code(), ErrorCode::InvalidArgument);
  EXPECT_NE(inf.message().find("infinite"), std::string::npos);

  const vf::Status big = validate_vector(std::vector<float>{1.0F, 1.0F, 1e20F}, 3);
  EXPECT_EQ(big.code(), ErrorCode::InvalidArgument);
  EXPECT_NE(big.message().find("magnitude"), std::string::npos);
}

TEST(Validation, ValidateBatch) {
  const std::vector<float> ok(2 * 4, 0.5F);
  EXPECT_TRUE(validate_batch(ok, 2, 4).ok());
  EXPECT_EQ(validate_batch(ok, 3, 4).code(), ErrorCode::DimensionMismatch);
  EXPECT_EQ(validate_batch(ok, 2, 3).code(), ErrorCode::DimensionMismatch);
  EXPECT_TRUE(validate_batch(std::vector<float>{}, 0, 4).ok());

  std::vector<float> bad(2 * 4, 0.5F);
  bad[6] = kNaN;  // row 1, component 2
  const vf::Status s = validate_batch(bad, 2, 4);
  EXPECT_EQ(s.code(), ErrorCode::InvalidArgument);
  EXPECT_NE(s.message().find("row 1 component 2"), std::string::npos) << s.message();
}

}  // namespace
