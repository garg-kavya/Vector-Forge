#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <vectorforge/status.hpp>

namespace {

using vf::ErrorCode;
using vf::Result;
using vf::Status;

TEST(Status, DefaultIsOk) {
  const Status s;
  EXPECT_TRUE(s.ok());
  EXPECT_EQ(s.code(), ErrorCode::Ok);
  EXPECT_TRUE(s.message().empty());
  EXPECT_EQ(s.to_string(), "OK");
  EXPECT_TRUE(Status::ok_status().ok());
}

TEST(Status, FactoriesSetCodeAndMessage) {
  const std::pair<Status, ErrorCode> cases[] = {
      {Status::invalid_argument("m"), ErrorCode::InvalidArgument},
      {Status::dimension_mismatch("m"), ErrorCode::DimensionMismatch},
      {Status::not_found("m"), ErrorCode::NotFound},
      {Status::already_exists("m"), ErrorCode::AlreadyExists},
      {Status::corrupt_data("m"), ErrorCode::CorruptData},
      {Status::unsupported_version("m"), ErrorCode::UnsupportedVersion},
      {Status::io_error("m"), ErrorCode::IoError},
      {Status::resource_exhausted("m"), ErrorCode::ResourceExhausted},
      {Status::failed_precondition("m"), ErrorCode::FailedPrecondition},
      {Status::unavailable("m"), ErrorCode::Unavailable},
      {Status::internal("m"), ErrorCode::Internal},
  };
  for (const auto& [status, code] : cases) {
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), code);
    EXPECT_EQ(status.message(), "m");
    EXPECT_NE(vf::to_string(code), "UNKNOWN");
  }
}

TEST(Status, ToStringIncludesCodeAndMessage) {
  EXPECT_EQ(Status::dimension_mismatch("expected 3, got 4").to_string(),
            "DIMENSION_MISMATCH: expected 3, got 4");
  EXPECT_EQ(Status(ErrorCode::NotFound, "").to_string(), "NOT_FOUND");
}

TEST(Status, Equality) {
  EXPECT_EQ(Status::not_found("a"), Status::not_found("a"));
  EXPECT_FALSE(Status::not_found("a") == Status::not_found("b"));
  EXPECT_FALSE(Status::not_found("a") == Status::internal("a"));
}

Status fails_when(bool fail) {
  if (fail) {
    return Status::io_error("boom");
  }
  return {};
}

Status propagate(bool fail, int& reached) {
  VF_RETURN_IF_ERROR(fails_when(fail));
  reached = 1;
  return {};
}

TEST(Status, ReturnIfErrorMacro) {
  int reached = 0;
  EXPECT_EQ(propagate(true, reached).code(), ErrorCode::IoError);
  EXPECT_EQ(reached, 0);
  EXPECT_TRUE(propagate(false, reached).ok());
  EXPECT_EQ(reached, 1);
}

TEST(Result, HoldsValue) {
  Result<int> r = 42;
  ASSERT_TRUE(r.has_value());
  EXPECT_TRUE(r.ok());
  EXPECT_TRUE(static_cast<bool>(r));
  EXPECT_EQ(r.value(), 42);
  EXPECT_EQ(*r, 42);
  EXPECT_TRUE(r.status().ok());
  EXPECT_TRUE(r.error().ok());
  r.value() = 7;
  EXPECT_EQ(r.value_or(0), 7);
}

TEST(Result, HoldsError) {
  const Result<std::string> r = Status::not_found("missing");
  ASSERT_FALSE(r.has_value());
  EXPECT_FALSE(static_cast<bool>(r));
  EXPECT_EQ(r.status().code(), ErrorCode::NotFound);
  EXPECT_EQ(r.error().message(), "missing");
  EXPECT_EQ(r.value_or("fallback"), "fallback");
}

TEST(Result, ConvertingConstruction) {
  const Result<std::uint32_t> small = 5;  // int literal -> uint32_t
  EXPECT_EQ(small.value(), 5U);
  const Result<std::string> text = "hello";  // const char* -> std::string
  EXPECT_EQ(text->size(), 5U);
}

TEST(Result, MoveOnlyPayload) {
  Result<std::unique_ptr<int>> r = std::make_unique<int>(9);
  ASSERT_TRUE(r.ok());
  std::unique_ptr<int> owned = std::move(r).value();
  ASSERT_NE(owned, nullptr);
  EXPECT_EQ(*owned, 9);
}

Result<int> parse_positive(int x) {
  if (x <= 0) {
    return Status::invalid_argument("not positive");
  }
  return x;
}

TEST(Result, FunctionReturn) {
  EXPECT_EQ(parse_positive(3).value(), 3);
  EXPECT_EQ(parse_positive(-1).status().code(), ErrorCode::InvalidArgument);
}

#if GTEST_HAS_DEATH_TEST
TEST(ResultDeathTest, ValueOnErrorTerminates) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const Result<int> r = Status::internal("bad");
  EXPECT_DEATH({ [[maybe_unused]] int v = r.value(); }, "INTERNAL: bad");
}

TEST(ResultDeathTest, ConstructionFromOkStatusTerminates) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH({ [[maybe_unused]] const Result<int> r = Status{}; }, "OK Status");
}
#endif

}  // namespace
