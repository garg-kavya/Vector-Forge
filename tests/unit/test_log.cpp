// Structured logger (src/util/log.hpp): level filtering, logfmt quoting, sink redirection.

#include <gtest/gtest.h>

#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "util/log.hpp"

namespace {

namespace vlog = vf::log;

class LogTest : public testing::Test {
 protected:
  void SetUp() override {
    vlog::set_sink([this](std::string_view line) {
      const std::lock_guard<std::mutex> lock(mutex_);
      lines_.emplace_back(line);
    });
    vlog::set_level(vlog::Level::Info);
  }
  void TearDown() override {
    vlog::set_sink(nullptr);
    vlog::set_level(vlog::Level::Info);
  }
  std::vector<std::string> lines() {
    const std::lock_guard<std::mutex> lock(mutex_);
    return lines_;
  }

 private:
  std::mutex mutex_;
  std::vector<std::string> lines_;
};

TEST_F(LogTest, FormatsFieldsAndQuotesUnsafeValues) {
  EXPECT_EQ(vlog::format(vlog::Level::Info, "request",
                         {{"method", "GET"}, {"status", 200}, {"ok", true}, {"ms", 1.5}}, false),
            "level=info event=request method=GET status=200 ok=true ms=1.5");
  EXPECT_EQ(vlog::format(vlog::Level::Warn, "x", {{"path", "/a b"}, {"q", "say \"hi\""}}, false),
            R"(level=warn event=x path="/a b" q="say \"hi\"")");
  // Newlines and control characters can never split or forge a line.
  const std::string escaped = vlog::format(vlog::Level::Error, "x", {{"v", "a\nb=c\x01"}}, false);
  EXPECT_EQ(escaped, R"(level=error event=x v="a\nb=c\x01")");
  EXPECT_EQ(escaped.find('\n'), std::string::npos);
}

TEST_F(LogTest, LevelsFilterAndParse) {
  vlog::write(vlog::Level::Debug, "hidden");
  vlog::write(vlog::Level::Info, "shown", {{"n", 1}});
  vlog::set_level(vlog::Level::Error);
  vlog::write(vlog::Level::Warn, "hidden");
  vlog::write(vlog::Level::Error, "shown2");
  vlog::set_level(vlog::Level::Off);
  vlog::write(vlog::Level::Error, "hidden");
  const auto out = lines();
  ASSERT_EQ(out.size(), 2U);
  EXPECT_EQ(out[0].rfind("ts=", 0), 0U);
  EXPECT_NE(out[0].find(" level=info event=shown n=1"), std::string::npos);
  EXPECT_NE(out[1].find("event=shown2"), std::string::npos);

  vlog::Level parsed = vlog::Level::Info;
  EXPECT_TRUE(vlog::parse_level("warn", parsed));
  EXPECT_EQ(parsed, vlog::Level::Warn);
  EXPECT_FALSE(vlog::parse_level("verbose", parsed));
  EXPECT_EQ(vlog::to_string(vlog::Level::Off), "off");
  EXPECT_FALSE(vlog::enabled(vlog::Level::Off));
}

TEST_F(LogTest, ConcurrentWritersProduceWholeLines) {
  std::vector<std::thread> threads;
  for (int t = 0; t < 4; ++t) {
    threads.emplace_back([t] {
      for (int i = 0; i < 200; ++i) {
        vlog::write(vlog::Level::Info, "tick", {{"thread", t}, {"i", i}});
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  const auto out = lines();
  ASSERT_EQ(out.size(), 800U);
  for (const std::string& line : out) {
    EXPECT_NE(line.find("event=tick thread="), std::string::npos) << line;
  }
}

}  // namespace
