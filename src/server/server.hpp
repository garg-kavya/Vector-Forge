#pragma once

// HTTP/JSON server over a Catalog (docs/DESIGN.md §13, docs/http-api.md).
//
// Lifecycle: start() binds and serves on a background thread; stop() turns readiness off, lets
// requests in progress finish (new requests get 503 meanwhile, up to the drain timeout), then
// closes the listener and joins the worker threads. Handlers use only the public Catalog and
// Collection API; a request holds a std::shared_ptr to its collection, so dropping a collection
// never frees it under a running request.
//
// Thread safety: start() and stop() must not race with each other; everything else is internal.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>

#include <vectorforge/catalog.hpp>
#include <vectorforge/status.hpp>

#include "server/limits.hpp"

namespace vf::server {

struct ServerConfig {
  std::string host = "127.0.0.1";
  int port = 8080;                  // 0: any free port (see Server::port())
  std::size_t http_threads = 0;     // 0: hardware concurrency
  std::size_t compute_threads = 0;  // pool for batch requests; 0: hardware concurrency
  Limits limits;
  std::string api_key;  // non-empty: require "Authorization: Bearer <api_key>" on /v1 routes
  std::chrono::seconds read_timeout{30};
  std::chrono::seconds write_timeout{30};
  std::chrono::seconds keep_alive_timeout{5};
  std::chrono::milliseconds drain_timeout{10000};
  bool access_log = false;  // one log line per request (vf::log, level info)
};

struct RouteInfo {
  std::string_view method;
  std::string_view path;  // OpenAPI form, e.g. "/v1/collections/{name}"
};

class Server {
 public:
  Server(Catalog& catalog, ServerConfig config);
  Server(const Server&) = delete;
  Server& operator=(const Server&) = delete;
  Server(Server&&) = delete;
  Server& operator=(Server&&) = delete;
  ~Server();  // stop()

  // Errors: IoError (cannot bind), FailedPrecondition (already started).
  [[nodiscard]] Status start();
  // Idempotent.
  void stop();

  [[nodiscard]] int port() const noexcept;
  [[nodiscard]] bool ready() const noexcept;

  // Every route the server registers (used to check docs/openapi.yaml).
  [[nodiscard]] static std::span<const RouteInfo> routes() noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vf::server
