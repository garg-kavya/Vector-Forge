// Server lifecycle (docs/DESIGN.md §13.5): bind errors, readiness, draining of requests in
// progress, dropping a collection used by running requests, and concurrent clients.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "http_fixture.hpp"

namespace {

using vf::test::HttpFixture;
using vf::test::json;

TEST(HttpShutdown, BindFailureAndDoubleStart) {
  HttpFixture f;
  vf::server::ServerConfig cfg;
  cfg.port = f.server().port();  // already taken
  vf::server::Server second(f.catalog(), cfg);
  EXPECT_EQ(second.start().code(), vf::ErrorCode::IoError);
  EXPECT_EQ(f.server().start().code(), vf::ErrorCode::FailedPrecondition);
}

TEST(HttpShutdown, StopDrainsRequestsInProgress) {
  HttpFixture f;
  f.create("big", 32, "l2", "flat");
  // Enough data that a batch search takes a noticeable time.
  const vf::test::ClusteredData data(1, 32, 8, 0.5F);
  const std::vector<float> rows = data.rows(1, 20000);
  std::vector<vf::ExternalId> ids(20000);
  for (std::size_t i = 0; i < ids.size(); ++i) {
    ids[i] = i;
  }
  ASSERT_TRUE(f.catalog().get("big").value()->add_batch(ids, rows).ok());
  json queries = json::array();
  const std::vector<float> q = data.rows(2, 1000);
  for (std::size_t i = 0; i < 1000; ++i) {
    queries.push_back(std::vector<float>(q.begin() + static_cast<std::ptrdiff_t>(i * 32),
                                         q.begin() + static_cast<std::ptrdiff_t>((i + 1) * 32)));
  }
  const std::string body = json({{"vectors", queries}, {"k", 10}}).dump();

  std::atomic<int> status{0};
  std::thread client([&] {
    httplib::Client c("127.0.0.1", f.server().port());
    c.set_read_timeout(std::chrono::seconds(60));
    auto res = c.Post("/v1/collections/big/search:batch", body, "application/json");
    status.store(res ? res->status : -1);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  f.server().stop();
  EXPECT_FALSE(f.server().ready());
  client.join();
  // Either the request was already running (and completed) or it arrived after readiness went
  // off (503); it never fails with a connection error.
  EXPECT_TRUE(status.load() == 200 || status.load() == 503) << status.load();

  httplib::Client late("127.0.0.1", f.server().port());
  late.set_connection_timeout(std::chrono::seconds(2));
  EXPECT_FALSE(late.Get("/healthz")) << "the listener is closed after stop()";
  f.server().stop();  // idempotent
}

TEST(HttpShutdown, DropWhileInUseKeepsTheCollectionAlive) {
  HttpFixture f;
  f.create("d", 2);
  std::shared_ptr<vf::Collection> held = f.catalog().get("d").value();
  ASSERT_TRUE(held->add(1, std::vector<float>{1, 2}).ok());
  auto dropped = f.client().Delete("/v1/collections/d");
  ASSERT_TRUE(dropped);
  EXPECT_EQ(dropped->status, 200);
  EXPECT_TRUE(held->get(1).ok()) << "the in-flight holder still works";
  // The name is busy until the last reference is gone.
  auto busy = f.post_json("/v1/collections", {{"name", "d"}, {"dim", 2}});
  ASSERT_TRUE(busy);
  EXPECT_EQ(busy->status, 503) << busy->body;
  held.reset();
  EXPECT_FALSE(std::filesystem::exists(f.dir().path() / "collections" / "d"));
  auto recreated = f.post_json("/v1/collections", {{"name", "d"}, {"dim", 3}});
  ASSERT_TRUE(recreated);
  EXPECT_EQ(recreated->status, 201) << recreated->body;
}

TEST(HttpShutdown, ConcurrentClients) {
  HttpFixture f;
  f.create("c", 4);
  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < 6; ++t) {
    threads.emplace_back([&, t] {
      httplib::Client c("127.0.0.1", f.server().port());
      for (int i = 0; i < 40; ++i) {
        const auto x = static_cast<float>(i);
        const json insert = {
            {"vectors", {{{"id", (t * 1000) + i}, {"vector", {x, x, x, static_cast<float>(t)}}}}}};
        auto a = c.Post("/v1/collections/c/vectors", insert.dump(), "application/json");
        auto s = c.Post("/v1/collections/c/search",
                        json({{"vector", {x, x, x, 0.0}}, {"k", 3}}).dump(), "application/json");
        if (!a || a->status != 200 || !s || s->status != 200) {
          ++failures;
        }
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  EXPECT_EQ(failures.load(), 0);
  EXPECT_EQ(f.catalog().get("c").value()->size(), 240U);
}

}  // namespace
