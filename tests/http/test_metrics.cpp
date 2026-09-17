// /metrics (src/server/metrics.hpp): exposition format, request counters, latency histograms,
// collection gauges, authentication; access log lines.

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "http_fixture.hpp"
#include "server/metrics.hpp"
#include "util/log.hpp"

namespace {

using vf::test::HttpFixture;
using vf::test::json;

// Parses "name{labels} value" lines; fails the test on anything that is not a comment or sample.
std::map<std::string, double> parse_samples(const std::string& text) {
  static const std::regex sample(
      R"(^([a-zA-Z_:][a-zA-Z0-9_:]*(\{.*\})?) (-?[0-9.eE+-]+|\+Inf|NaN)$)");
  std::map<std::string, double> out;
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line.starts_with("# HELP ") || line.starts_with("# TYPE ")) {
      continue;
    }
    std::smatch m;
    EXPECT_TRUE(std::regex_match(line, m, sample)) << "bad line: " << line;
    if (!m.empty()) {
      out[m[1].str()] = std::stod(m[3].str());
    }
  }
  return out;
}

TEST(HttpMetrics, CountersHistogramsAndGauges) {
  HttpFixture f;
  f.create("m", 2);
  for (int i = 0; i < 3; ++i) {
    ASSERT_EQ(f.client().Get("/healthz")->status, 200);
  }
  ASSERT_EQ(
      f.post_json("/v1/collections/m/vectors",
                  {{"vectors", {{{"id", 1}, {"vector", {1, 2}}}, {{"id", 2}, {"vector", {3, 4}}}}}})
          ->status,
      200);
  ASSERT_EQ(f.client().Delete("/v1/collections/m/vectors/2")->status, 200);
  ASSERT_EQ(f.post_json("/v1/collections/m/search", {{"vector", {1, 2, 3}}})->status, 400);
  ASSERT_EQ(f.client().Get("/nope")->status, 404);

  auto res = f.client().Get("/metrics");
  ASSERT_TRUE(res);
  ASSERT_EQ(res->status, 200);
  EXPECT_EQ(res->get_header_value("Content-Type").rfind("text/plain; version=0.0.4", 0), 0U);
  const std::map<std::string, double> s = parse_samples(res->body);

  EXPECT_EQ(s.at(R"(vectorforge_http_requests_total{method="GET",route="/healthz",status="2xx"})"),
            3);
  EXPECT_EQ(
      s.at(
          R"(vectorforge_http_requests_total{method="POST",route="/v1/collections/{name}/search",status="4xx"})"),
      1);
  EXPECT_EQ(
      s.at(
          R"(vectorforge_http_requests_total{method="DELETE",route="/v1/collections/{name}/vectors/{id}",status="2xx"})"),
      1);
  EXPECT_EQ(s.at(R"(vectorforge_http_requests_total{method="",route="unmatched",status="4xx"})"),
            1);
  const std::string h = R"(vectorforge_http_request_duration_seconds)";
  const std::string l = R"(method="GET",route="/healthz")";
  EXPECT_EQ(s.at(h + "_count{" + l + "}"), 3);
  EXPECT_EQ(s.at(h + "_bucket{" + l + R"(,le="+Inf"})"), 3);
  EXPECT_LE(s.at(h + "_bucket{" + l + R"(,le="0.001"})"),
            s.at(h + "_bucket{" + l + R"(,le="0.01"})"));
  EXPECT_GT(s.at(h + "_sum{" + l + "}"), 0.0);

  EXPECT_EQ(s.at("vectorforge_collections"), 1);
  EXPECT_EQ(s.at(R"(vectorforge_collection_vectors{collection="m"})"), 1);
  EXPECT_EQ(s.at(R"(vectorforge_collection_deleted_vectors{collection="m"})"), 1);
  EXPECT_GT(s.at(R"(vectorforge_collection_memory_bytes{collection="m"})"), 0);
  EXPECT_EQ(s.at(R"(vectorforge_collection_snapshot_generation{collection="m"})"), 0);
  EXPECT_EQ(s.at("vectorforge_http_in_flight_requests"), 1);  // the scrape itself
  EXPECT_GE(s.at("vectorforge_uptime_seconds"), 0.0);
  bool build_info = false;
  for (const auto& [k, v] : s) {
    build_info = build_info || (k.starts_with("vectorforge_build_info{") && v == 1);
  }
  EXPECT_TRUE(build_info);

  // The scrape is counted by the next scrape.
  const auto again = parse_samples(f.client().Get("/metrics")->body);
  EXPECT_EQ(
      again.at(R"(vectorforge_http_requests_total{method="GET",route="/metrics",status="2xx"})"),
      1);
}

TEST(HttpMetrics, RequiresTheBearerTokenWhenConfigured) {
  vf::server::ServerConfig cfg;
  cfg.api_key = "k";
  HttpFixture f(cfg, "http_metrics_auth");
  EXPECT_EQ(f.client().Get("/metrics")->status, 401);
  EXPECT_EQ(f.client().Get("/metrics", httplib::Headers{{"Authorization", "Bearer k"}})->status,
            200);
}

TEST(HttpMetrics, HistogramBucketsAndLabelEscaping) {
  vf::server::LatencyHistogram h;
  h.observe(0.00004);  // first bucket
  h.observe(0.0003);   // le 0.0005
  h.observe(20.0);     // +Inf only
  const auto c = h.cumulative();
  EXPECT_EQ(c.front(), 1U);
  EXPECT_EQ(c[3], 2U);
  EXPECT_EQ(c[c.size() - 2], 2U);
  EXPECT_EQ(c.back(), 3U);
  EXPECT_EQ(h.count(), 3U);
  EXPECT_NEAR(h.sum_seconds(), 20.00034, 1e-6);
  EXPECT_EQ(vf::server::escape_label("a\"b\\c\nd"), R"(a\"b\\c\nd)");
}

TEST(HttpMetrics, AccessLog) {
  std::mutex mutex;
  std::vector<std::string> lines;
  vf::log::set_sink([&](std::string_view line) {
    const std::lock_guard<std::mutex> lock(mutex);
    lines.emplace_back(line);
  });
  {
    vf::server::ServerConfig cfg;
    cfg.access_log = true;
    HttpFixture f(cfg, "http_access_log");
    ASSERT_EQ(f.client().Get("/healthz", httplib::Headers{{"X-Request-Id", "req-7"}})->status, 200);
    ASSERT_EQ(f.client().Get("/v1/collections/x%20y")->status, 404);
  }
  vf::log::set_sink(nullptr);
  const std::lock_guard<std::mutex> lock(mutex);
  ASSERT_GE(lines.size(), 2U);
  EXPECT_NE(lines[0].find("event=request method=GET path=/healthz status=200"), std::string::npos)
      << lines[0];
  EXPECT_NE(lines[0].find("id=req-7"), std::string::npos);
  EXPECT_NE(lines[1].find(R"(path="/v1/collections/x y" status=404)"), std::string::npos)
      << lines[1];
}

}  // namespace
