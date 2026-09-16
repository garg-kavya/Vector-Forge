// Request limits (docs/DESIGN.md §13.5): body size (413), batch size, k, ef_search and dim caps
// (422), all reported with JSON error bodies.

#include <gtest/gtest.h>

#include <string>

#include "http_fixture.hpp"

namespace {

using vf::test::HttpFixture;
using vf::test::json;

vf::server::ServerConfig small_limits() {
  vf::server::ServerConfig cfg;
  cfg.limits.max_body_bytes = 4096;
  cfg.limits.max_batch = 3;
  cfg.limits.max_k = 5;
  cfg.limits.max_ef = 50;
  cfg.limits.max_dim = 16;
  return cfg;
}

void expect_error(const httplib::Result& res, int status, const std::string& code) {
  ASSERT_TRUE(res);
  EXPECT_EQ(res->status, status) << res->body;
  const json body = HttpFixture::body_of(res);
  ASSERT_TRUE(body.is_object()) << res->body;
  EXPECT_EQ(body["error"]["code"], code) << res->body;
}

TEST(HttpLimits, BodySize) {
  HttpFixture f(small_limits(), "http_limits");
  f.create("l", 2);
  std::string big = R"({"vector": [1, 2], "k": 1, "pad": ")" + std::string(5000, 'x') + "\"}";
  expect_error(f.client().Post("/v1/collections/l/search", big, "application/json"), 413,
               "PAYLOAD_TOO_LARGE");
  // The server keeps working afterwards.
  auto ok =
      f.client().Post("/v1/collections/l/search", R"({"vector": [1, 2]})", "application/json");
  ASSERT_TRUE(ok);
  EXPECT_EQ(ok->status, 200);
}

TEST(HttpLimits, BatchKEfAndDim) {
  HttpFixture f(small_limits(), "http_limits");
  f.create("l", 2);
  const json four = {{"vectors",
                      {{{"id", 1}, {"vector", {1, 2}}},
                       {{"id", 2}, {"vector", {1, 2}}},
                       {{"id", 3}, {"vector", {1, 2}}},
                       {{"id", 4}, {"vector", {1, 2}}}}}};
  expect_error(f.post_json("/v1/collections/l/vectors", four), 422, "LIMIT_EXCEEDED");
  EXPECT_EQ(f.catalog().get("l").value()->size(), 0U) << "a rejected batch inserts nothing";
  expect_error(f.post_json("/v1/collections/l/search:batch",
                           {{"vectors", {{1, 2}, {1, 2}, {1, 2}, {1, 2}}}}),
               422, "LIMIT_EXCEEDED");
  expect_error(f.post_json("/v1/collections/l/search", {{"vector", {1, 2}}, {"k", 6}}), 422,
               "LIMIT_EXCEEDED");
  expect_error(f.post_json("/v1/collections/l/search", {{"vector", {1, 2}}, {"ef_search", 51}}),
               422, "LIMIT_EXCEEDED");
  expect_error(f.post_json("/v1/collections", {{"name", "big"}, {"dim", 17}}), 422,
               "LIMIT_EXCEEDED");

  auto at_limit =
      f.post_json("/v1/collections/l/search", {{"vector", {1, 2}}, {"k", 5}, {"ef_search", 50}});
  ASSERT_TRUE(at_limit);
  EXPECT_EQ(at_limit->status, 200) << at_limit->body;
}

TEST(HttpLimits, DeeplyNestedJsonIsRejectedBeforeParsing) {
  HttpFixture f(small_limits(), "http_limits");
  f.create("l", 2);
  const std::string nested = std::string(2000, '[') + std::string(2000, ']');
  // 4 000 bytes fit the body limit; the nesting check rejects it.
  expect_error(f.client().Post("/v1/collections/l/search", nested, "application/json"), 400,
               "MALFORMED_JSON");
}

}  // namespace
