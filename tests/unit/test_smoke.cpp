#include <gtest/gtest.h>

#include <string>

#include <vectorforge/version.hpp>

#include "core/build_info.hpp"

namespace {

TEST(Smoke, VersionIsPopulated) {
  EXPECT_FALSE(vf::kVersion.empty());
  EXPECT_FALSE(vf::kGitSha.empty());
  EXPECT_GE(vf::kVersionMajor, 0);
}

TEST(Smoke, BuildSummaryMentionsVersion) {
  const std::string summary = vf::detail::build_summary();
  EXPECT_NE(summary.find(std::string(vf::kVersion)), std::string::npos);
}

}  // namespace
