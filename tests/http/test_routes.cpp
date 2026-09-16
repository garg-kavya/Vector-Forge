// HTTP routes (docs/http-api.md): happy paths of every route, results against the in-process API,
// and the error mapping of docs/DESIGN.md §13.4.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "http_fixture.hpp"

namespace {

using vf::test::HttpFixture;
using vf::test::json;

constexpr std::uint32_t kDim = 4;

json vectors_body(std::uint64_t first_id, std::size_t count, bool upsert = false) {
  json vectors = json::array();
  for (std::size_t i = 0; i < count; ++i) {
    const auto x = static_cast<float>(first_id + i);
    vectors.push_back({{"id", first_id + i}, {"vector", {x, x + 0.5F, -x, 1.0F}}});
  }
  return {{"upsert", upsert}, {"vectors", vectors}};
}

std::string bulk_body(std::uint32_t dim, const std::vector<std::uint64_t>& ids,
                      const std::vector<float>& rows) {
  std::string body = "VFB1";
  auto put = [&body](std::uint64_t value, int bytes) {
    for (int i = 0; i < bytes; ++i) {
      body.push_back(static_cast<char>((value >> (8 * i)) & 0xFFU));
    }
  };
  put(dim, 4);
  put(ids.size(), 8);
  for (const std::uint64_t id : ids) {
    put(id, 8);
  }
  for (const float f : rows) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    put(bits, 4);
  }
  return body;
}

TEST(HttpRoutes, HealthReadinessAndStatus) {
  HttpFixture f;
  auto health = f.client().Get("/healthz");
  ASSERT_TRUE(health);
  EXPECT_EQ(health->status, 200);
  EXPECT_FALSE(health->get_header_value("X-Request-Id").empty());
  auto ready = f.client().Get("/readyz");
  ASSERT_TRUE(ready);
  EXPECT_EQ(ready->status, 200);
  auto status = f.client().Get("/v1/status");
  ASSERT_TRUE(status);
  ASSERT_EQ(status->status, 200);
  const json body = HttpFixture::body_of(status);
  EXPECT_TRUE(body.contains("version"));
  EXPECT_TRUE(body.contains("simd"));
  EXPECT_EQ(body["collections"], 0);
  EXPECT_EQ(status->get_header_value("Content-Type"), "application/json");
}

TEST(HttpRoutes, RequestIdIsEchoedOrGenerated) {
  HttpFixture f;
  auto echoed = f.client().Get("/healthz", httplib::Headers{{"X-Request-Id", "abc-123"}});
  ASSERT_TRUE(echoed);
  EXPECT_EQ(echoed->get_header_value("X-Request-Id"), "abc-123");
  auto a = f.client().Get("/healthz", httplib::Headers{{"X-Request-Id", std::string(200, 'x')}});
  auto b = f.client().Get("/healthz");
  ASSERT_TRUE(a && b);
  EXPECT_NE(a->get_header_value("X-Request-Id"), std::string(200, 'x'));
  EXPECT_NE(a->get_header_value("X-Request-Id"), b->get_header_value("X-Request-Id"));
}

TEST(HttpRoutes, CollectionLifecycle) {
  HttpFixture f;
  auto created = f.post_json(
      "/v1/collections",
      {{"name", "docs"},
       {"dim", kDim},
       {"metric", "cosine"},
       {"index",
        {{"type", "hnsw"}, {"M", 8}, {"ef_construction", 64}, {"ef_search", 32}, {"seed", 42}}}});
  ASSERT_TRUE(created);
  ASSERT_EQ(created->status, 201) << created->body;
  json body = HttpFixture::body_of(created);
  EXPECT_EQ(body["name"], "docs");
  EXPECT_EQ(body["dim"], kDim);
  EXPECT_EQ(body["metric"], "cosine");
  EXPECT_EQ(body["normalize"], true);
  EXPECT_EQ(body["index"]["M"], 8);
  EXPECT_EQ(body["index"]["seed"], 42);
  EXPECT_EQ(body["count"], 0);

  auto duplicate = f.post_json("/v1/collections", {{"name", "docs"}, {"dim", kDim}});
  ASSERT_TRUE(duplicate);
  EXPECT_EQ(duplicate->status, 409);
  EXPECT_EQ(HttpFixture::body_of(duplicate)["error"]["code"], "ALREADY_EXISTS");

  f.create("flat", 3, "l2", "flat");
  auto list = f.client().Get("/v1/collections");
  ASSERT_TRUE(list);
  body = HttpFixture::body_of(list);
  ASSERT_EQ(body["collections"].size(), 2U);
  EXPECT_EQ(body["collections"][0]["name"], "docs");
  EXPECT_EQ(body["collections"][1]["index"]["type"], "flat");

  auto one = f.client().Get("/v1/collections/flat");
  ASSERT_TRUE(one);
  EXPECT_EQ(one->status, 200);
  EXPECT_EQ(HttpFixture::body_of(one)["dim"], 3);

  auto dropped = f.client().Delete("/v1/collections/flat");
  ASSERT_TRUE(dropped);
  EXPECT_EQ(dropped->status, 200);
  auto gone = f.client().Get("/v1/collections/flat");
  ASSERT_TRUE(gone);
  EXPECT_EQ(gone->status, 404);
  EXPECT_EQ(HttpFixture::body_of(gone)["error"]["code"], "NOT_FOUND");
  auto again = f.client().Delete("/v1/collections/flat");
  ASSERT_TRUE(again);
  EXPECT_EQ(again->status, 404);
}

TEST(HttpRoutes, InsertGetSearchDeleteAgreeWithTheLibrary) {
  HttpFixture f;
  f.create("v", kDim, "l2", "flat");
  auto inserted = f.post_json("/v1/collections/v/vectors", vectors_body(0, 20));
  ASSERT_TRUE(inserted);
  ASSERT_EQ(inserted->status, 200) << inserted->body;
  EXPECT_EQ(HttpFixture::body_of(inserted)["inserted"], 20);

  auto fetched = f.client().Get("/v1/collections/v/vectors/7");
  ASSERT_TRUE(fetched);
  ASSERT_EQ(fetched->status, 200);
  json body = HttpFixture::body_of(fetched);
  EXPECT_EQ(body["id"], 7);
  EXPECT_EQ(body["vector"], json({7.0, 7.5, -7.0, 1.0}));

  const std::vector<float> query{5.1F, 5.6F, -5.1F, 1.0F};
  auto searched =
      f.post_json("/v1/collections/v/search", {{"vector", query}, {"k", 3}, {"ef_search", 16}});
  ASSERT_TRUE(searched);
  ASSERT_EQ(searched->status, 200) << searched->body;
  body = HttpFixture::body_of(searched);
  ASSERT_TRUE(body.contains("took_ms"));
  const auto c = f.catalog().get("v").value();
  vf::SearchParams p;
  p.k = 3;
  const auto expected = c->search(query, p).value();
  ASSERT_EQ(body["results"].size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(body["results"][i]["id"], expected[i].id);
    EXPECT_FLOAT_EQ(body["results"][i]["distance"].get<float>(), expected[i].distance);
  }

  auto batch = f.post_json("/v1/collections/v/search:batch",
                           {{"vectors", {query, {0.0, 0.5, 0.0, 1.0}}}, {"k", 2}});
  ASSERT_TRUE(batch);
  ASSERT_EQ(batch->status, 200) << batch->body;
  body = HttpFixture::body_of(batch);
  ASSERT_EQ(body["results"].size(), 2U);
  EXPECT_EQ(body["results"][0][0]["id"], expected[0].id);
  EXPECT_EQ(body["results"][1][0]["id"], 0);

  auto removed = f.client().Delete("/v1/collections/v/vectors/5");
  ASSERT_TRUE(removed);
  EXPECT_EQ(removed->status, 200);
  auto missing = f.client().Get("/v1/collections/v/vectors/5");
  ASSERT_TRUE(missing);
  EXPECT_EQ(missing->status, 404);
  auto after = f.post_json("/v1/collections/v/search", {{"vector", query}, {"k", 1}});
  ASSERT_TRUE(after);
  EXPECT_NE(HttpFixture::body_of(after)["results"][0]["id"], 5);

  auto stats = f.client().Get("/v1/collections/v/stats");
  ASSERT_TRUE(stats);
  ASSERT_EQ(stats->status, 200);
  body = HttpFixture::body_of(stats);
  EXPECT_EQ(body["count"], 19);
  EXPECT_EQ(body["deleted"], 1);
  EXPECT_GT(body["memory"]["total_bytes"].get<std::uint64_t>(), 0U);

  auto compacted = f.client().Post("/v1/collections/v/compact");
  ASSERT_TRUE(compacted);
  ASSERT_EQ(compacted->status, 200) << compacted->body;
  EXPECT_EQ(HttpFixture::body_of(compacted)["removed_rows"], 1);

  auto upserted = f.post_json("/v1/collections/v/vectors", vectors_body(0, 2, true));
  ASSERT_TRUE(upserted);
  EXPECT_EQ(upserted->status, 200);
  auto conflict = f.post_json("/v1/collections/v/vectors", vectors_body(0, 1));
  ASSERT_TRUE(conflict);
  EXPECT_EQ(conflict->status, 409);
}

TEST(HttpRoutes, BulkInsert) {
  HttpFixture f;
  f.create("b", 2);
  const std::vector<std::uint64_t> ids{10, 11, 12};
  const std::vector<float> rows{1, 2, 3, 4, 5, 6};
  auto ok = f.client().Post("/v1/collections/b/vectors:bulk", bulk_body(2, ids, rows),
                            "application/octet-stream");
  ASSERT_TRUE(ok);
  ASSERT_EQ(ok->status, 200) << ok->body;
  EXPECT_EQ(HttpFixture::body_of(ok)["inserted"], 3);
  auto v = f.client().Get("/v1/collections/b/vectors/12");
  ASSERT_TRUE(v);
  EXPECT_EQ(HttpFixture::body_of(v)["vector"], json({5.0, 6.0}));

  auto conflict = f.client().Post("/v1/collections/b/vectors:bulk", bulk_body(2, ids, rows),
                                  "application/octet-stream");
  ASSERT_TRUE(conflict);
  EXPECT_EQ(conflict->status, 409);
  auto upsert = f.client().Post("/v1/collections/b/vectors:bulk?upsert=true",
                                bulk_body(2, ids, rows), "application/octet-stream");
  ASSERT_TRUE(upsert);
  EXPECT_EQ(upsert->status, 200);

  auto wrong_dim = f.client().Post("/v1/collections/b/vectors:bulk", bulk_body(3, {1}, {1, 2, 3}),
                                   "application/octet-stream");
  ASSERT_TRUE(wrong_dim);
  EXPECT_EQ(wrong_dim->status, 400);
  EXPECT_EQ(HttpFixture::body_of(wrong_dim)["error"]["code"], "DIMENSION_MISMATCH");
  std::string truncated = bulk_body(2, ids, rows);
  truncated.pop_back();
  auto short_body =
      f.client().Post("/v1/collections/b/vectors:bulk", truncated, "application/octet-stream");
  ASSERT_TRUE(short_body);
  EXPECT_EQ(short_body->status, 400);
  auto as_json = f.client().Post("/v1/collections/b/vectors:bulk", bulk_body(2, ids, rows),
                                 "application/json");
  ASSERT_TRUE(as_json);
  EXPECT_EQ(as_json->status, 415);
}

TEST(HttpRoutes, SnapshotSurvivesRestart) {
  vf::test::ScopedTempDir dir("http_restart");
  {
    auto catalog = vf::Catalog::open(dir.path()).value();
    vf::server::ServerConfig cfg;
    cfg.port = 0;
    vf::server::Server server(*catalog, cfg);
    ASSERT_TRUE(server.start().ok());
    httplib::Client client("127.0.0.1", server.port());
    ASSERT_EQ(
        client.Post("/v1/collections", R"({"name": "keep", "dim": 4})", "application/json")->status,
        201);
    ASSERT_EQ(
        client.Post("/v1/collections/keep/vectors", vectors_body(0, 50).dump(), "application/json")
            ->status,
        200);
    auto snap = client.Post("/v1/collections/keep/snapshot");
    ASSERT_TRUE(snap);
    ASSERT_EQ(snap->status, 200) << snap->body;
    json body = HttpFixture::body_of(snap);
    EXPECT_EQ(body["generation"], 1);
    EXPECT_GT(body["bytes"].get<std::uint64_t>(), 0U);
    server.stop();
  }
  auto catalog = vf::Catalog::open(dir.path()).value();
  vf::server::ServerConfig cfg;
  cfg.port = 0;
  vf::server::Server server(*catalog, cfg);
  ASSERT_TRUE(server.start().ok());
  httplib::Client client("127.0.0.1", server.port());
  auto v = client.Get("/v1/collections/keep/vectors/49");
  ASSERT_TRUE(v);
  ASSERT_EQ(v->status, 200);
  EXPECT_EQ(HttpFixture::body_of(v)["vector"], json({49.0, 49.5, -49.0, 1.0}));
  auto found = client.Post("/v1/collections/keep/search",
                           json({{"vector", {49.0, 49.5, -49.0, 1.0}}, {"k", 1}}).dump(),
                           "application/json");
  ASSERT_TRUE(found);
  EXPECT_EQ(HttpFixture::body_of(found)["results"][0]["id"], 49);
}

TEST(HttpRoutes, ErrorMapping) {
  HttpFixture f;
  f.create("e", kDim);
  auto expect_error = [](const httplib::Result& res, int status, const std::string& code) {
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, status) << res->body;
    const json body = HttpFixture::body_of(res);
    ASSERT_TRUE(body.contains("error")) << res->body;
    EXPECT_EQ(body["error"]["code"], code) << res->body;
    EXPECT_FALSE(body["error"]["message"].get<std::string>().empty());
  };
  auto post_raw = [&f](const std::string& path, const std::string& body) {
    return f.client().Post(path, body, "application/json");
  };
  expect_error(post_raw("/v1/collections/e/search", "{not json"), 400, "MALFORMED_JSON");
  expect_error(post_raw("/v1/collections/e/search", std::string(20, '[') + std::string(20, ']')),
               400, "MALFORMED_JSON");
  expect_error(post_raw("/v1/collections/e/search", "[1, 2]"), 400, "INVALID_ARGUMENT");
  expect_error(post_raw("/v1/collections/e/search", R"({"vector": [1, 2, 3, 4], "x": 1})"), 400,
               "INVALID_ARGUMENT");
  expect_error(post_raw("/v1/collections/e/search", R"({"vector": [1, 2, 3]})"), 400,
               "DIMENSION_MISMATCH");
  expect_error(post_raw("/v1/collections/e/search", R"({"vector": [1, 2, 3, "a"]})"), 400,
               "INVALID_ARGUMENT");
  expect_error(post_raw("/v1/collections/e/search", R"({"vector": [1, 2, 3, 1e39]})"), 400,
               "INVALID_ARGUMENT");
  expect_error(post_raw("/v1/collections/e/search", R"({"vector": [1, 2, 3, 4], "k": 0})"), 400,
               "INVALID_ARGUMENT");
  expect_error(post_raw("/v1/collections/e/search", R"({"vector": [1, 2, 3, 4], "k": 1.5})"), 400,
               "INVALID_ARGUMENT");
  expect_error(
      post_raw("/v1/collections/e/vectors", R"({"vectors": [{"id": -1, "vector": [1,2,3,4]}]})"),
      400, "INVALID_ARGUMENT");
  expect_error(post_raw("/v1/collections/e/vectors", R"({"vectors": [{"id": 1}]})"), 400,
               "INVALID_ARGUMENT");
  expect_error(post_raw("/v1/collections/nope/search", R"({"vector": [1, 2, 3, 4]})"), 404,
               "NOT_FOUND");
  expect_error(f.client().Get("/v1/collections/e/vectors/99"), 404, "NOT_FOUND");
  expect_error(f.client().Get("/v1/collections/e/vectors/18446744073709551615"), 400,
               "INVALID_ARGUMENT");
  expect_error(f.client().Get("/v1/nothing"), 404, "NOT_FOUND");
  expect_error(f.client().Get("/v1/collections/bad.name"), 404, "NOT_FOUND");
  expect_error(
      f.client().Post("/v1/collections/e/search", R"({"vector": [1,2,3,4]})", "text/plain"), 415,
      "UNSUPPORTED_MEDIA_TYPE");
  expect_error(post_raw("/v1/collections", R"({"name": "../x", "dim": 4})"), 400,
               "INVALID_ARGUMENT");
  expect_error(post_raw("/v1/collections", R"({"name": "x", "dim": 0})"), 400, "INVALID_ARGUMENT");
  expect_error(post_raw("/v1/collections", R"({"name": "x", "dim": 4, "metric": "hamming"})"), 400,
               "INVALID_ARGUMENT");
  expect_error(post_raw("/v1/collections", R"({"name": "x", "dim": 4, "index": {"M": 1}})"), 400,
               "INVALID_ARGUMENT");
  expect_error(post_raw("/v1/collections", R"({"name": "x", "dim": 4, "index": {"foo": 1}})"), 400,
               "INVALID_ARGUMENT");
  expect_error(f.client().Delete("/v1/collections/e/vectors/12345"), 404, "NOT_FOUND");
}

TEST(HttpRoutes, BearerAuthentication) {
  vf::server::ServerConfig cfg;
  cfg.api_key = "s3cret";
  HttpFixture f(cfg, "http_auth");
  auto open_route = f.client().Get("/healthz");
  ASSERT_TRUE(open_route);
  EXPECT_EQ(open_route->status, 200);
  auto missing = f.client().Get("/v1/collections");
  ASSERT_TRUE(missing);
  EXPECT_EQ(missing->status, 401);
  EXPECT_EQ(HttpFixture::body_of(missing)["error"]["code"], "UNAUTHORIZED");
  auto wrong =
      f.client().Get("/v1/collections", httplib::Headers{{"Authorization", "Bearer s3creT"}});
  ASSERT_TRUE(wrong);
  EXPECT_EQ(wrong->status, 401);
  auto right =
      f.client().Get("/v1/collections", httplib::Headers{{"Authorization", "Bearer s3cret"}});
  ASSERT_TRUE(right);
  EXPECT_EQ(right->status, 200);
}

TEST(HttpRoutes, RouteTableMatchesRegisteredRoutes) {
  HttpFixture f;
  f.create("r", 2);
  // Every documented route answers with something other than "no route" (404 from the router
  // has no "error.code" other than NOT_FOUND with the "no route" message).
  for (const vf::server::RouteInfo& route : vf::server::Server::routes()) {
    std::string path(route.path);
    auto replace = [&path](const std::string& from, const std::string& to) {
      if (const auto pos = path.find(from); pos != std::string::npos) {
        path.replace(pos, from.size(), to);
      }
    };
    replace("{name}", "r");
    replace("{id}", "1");
    httplib::Result res;
    if (route.method == "GET") {
      res = f.client().Get(path);
    } else if (route.method == "DELETE") {
      res = f.client().Delete(path);
    } else {
      res = f.client().Post(path, "{}", "application/json");
    }
    ASSERT_TRUE(res) << path;
    const json body = HttpFixture::body_of(res);
    const bool no_route = body.contains("error") &&
                          body["error"]["message"].get<std::string>().rfind("no route", 0) == 0;
    EXPECT_FALSE(no_route) << route.method << " " << path;
    if (route.method == "DELETE" && route.path == "/v1/collections/{name}") {
      f.create("r", 2);  // keep later routes meaningful
    }
  }
}

}  // namespace
