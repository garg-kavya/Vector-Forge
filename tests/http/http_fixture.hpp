#pragma once

// A server on a free loopback port over a catalog in a temporary directory, and a client for it.

#include <gtest/gtest.h>

#include <httplib.h>
#include <memory>
#include <string>

#include <vectorforge/catalog.hpp>

#include "server/server.hpp"
#include "support/test_data.hpp"

#include <nlohmann/json.hpp>

namespace vf::test {

using json = nlohmann::json;

class HttpFixture {
 public:
  explicit HttpFixture(server::ServerConfig config = {}, const std::string& tag = "http")
      : dir_(tag) {
    catalog_ = Catalog::open(dir_.path()).value();
    config.host = "127.0.0.1";
    config.port = 0;
    if (config.http_threads == 0) {
      config.http_threads = 4;
    }
    if (config.compute_threads == 0) {
      config.compute_threads = 2;
    }
    server_ = std::make_unique<server::Server>(*catalog_, config);
    const Status st = server_->start();
    EXPECT_TRUE(st.ok()) << st.to_string();
    client_ = std::make_unique<httplib::Client>("127.0.0.1", server_->port());
    client_->set_read_timeout(std::chrono::seconds(30));
  }

  [[nodiscard]] httplib::Client& client() { return *client_; }
  [[nodiscard]] server::Server& server() { return *server_; }
  [[nodiscard]] Catalog& catalog() { return *catalog_; }
  [[nodiscard]] const ScopedTempDir& dir() const { return dir_; }

  httplib::Result post_json(const std::string& path, const json& body,
                            const httplib::Headers& headers = {}) {
    return client_->Post(path, headers, body.dump(), "application/json");
  }

  // Creates an HNSW (or Flat) collection with `dim` dimensions; expects success.
  void create(const std::string& name, std::uint32_t dim, const std::string& metric = "l2",
              const std::string& index = "hnsw") {
    const auto res =
        post_json("/v1/collections",
                  {{"name", name}, {"dim", dim}, {"metric", metric}, {"index", {{"type", index}}}});
    ASSERT_TRUE(res);
    ASSERT_EQ(res->status, 201) << res->body;
  }

  static json body_of(const httplib::Result& res) { return json::parse(res->body, nullptr, false); }

 private:
  ScopedTempDir dir_;
  std::unique_ptr<Catalog> catalog_;
  std::unique_ptr<server::Server> server_;
  std::unique_ptr<httplib::Client> client_;
};

}  // namespace vf::test
