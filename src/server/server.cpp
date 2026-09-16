#include "server/server.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <httplib.h>
#include <limits>
#include <mutex>
#include <new>
#include <random>
#include <semaphore>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include <vectorforge/simd.hpp>
#include <vectorforge/thread_pool.hpp>
#include <vectorforge/version.hpp>

#include "server/json_codec.hpp"

namespace vf::server {

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::string_view kJson = "application/json";
constexpr std::string_view kName = R"(([A-Za-z0-9_-]{1,64}))";
constexpr std::size_t kMaxRequestIdBytes = 128;
// Batch requests allowed to use the compute pool at the same time.
constexpr std::ptrdiff_t kBatchSlots = 4;

constexpr std::array kRoutes{
    RouteInfo{"GET", "/healthz"},
    RouteInfo{"GET", "/readyz"},
    RouteInfo{"GET", "/v1/status"},
    RouteInfo{"POST", "/v1/collections"},
    RouteInfo{"GET", "/v1/collections"},
    RouteInfo{"GET", "/v1/collections/{name}"},
    RouteInfo{"DELETE", "/v1/collections/{name}"},
    RouteInfo{"POST", "/v1/collections/{name}/vectors"},
    RouteInfo{"POST", "/v1/collections/{name}/vectors:bulk"},
    RouteInfo{"GET", "/v1/collections/{name}/vectors/{id}"},
    RouteInfo{"DELETE", "/v1/collections/{name}/vectors/{id}"},
    RouteInfo{"POST", "/v1/collections/{name}/search"},
    RouteInfo{"POST", "/v1/collections/{name}/search:batch"},
    RouteInfo{"GET", "/v1/collections/{name}/stats"},
    RouteInfo{"POST", "/v1/collections/{name}/snapshot"},
    RouteInfo{"POST", "/v1/collections/{name}/compact"},
};

double ms_since(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

bool constant_time_equal(std::string_view a, std::string_view b) noexcept {
  // The length is not secret; the comparison of equal-length strings does not exit early.
  if (a.size() != b.size()) {
    return false;
  }
  unsigned int diff = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    diff |= static_cast<unsigned int>(static_cast<unsigned char>(a[i]) ^
                                      static_cast<unsigned char>(b[i]));
  }
  return diff == 0;
}

bool is_valid_request_id(std::string_view id) noexcept {
  return !id.empty() && id.size() <= kMaxRequestIdBytes &&
         std::all_of(id.begin(), id.end(), [](char c) { return c > ' ' && c < 0x7F; });
}

void send_error(httplib::Response& res, const ApiError& error) {
  res.status = error.http_status;
  res.set_content(encode_error(error), std::string(kJson));
}

void send_status(httplib::Response& res, const Status& status) {
  send_error(res, to_api_error(status));
}

void send_json(httplib::Response& res, int http_status, std::string body) {
  res.status = http_status;
  res.set_content(std::move(body), std::string(kJson));
}

bool parse_id(const std::string& text, ExternalId& out) {
  const auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), out);
  return ec == std::errc{} && ptr == text.data() + text.size() && out != kInvalidExternalId;
}

}  // namespace

struct Server::Impl {
  Impl(Catalog& c, ServerConfig cfg) : catalog(c), config(std::move(cfg)) {
    const std::size_t hw = std::max<std::size_t>(std::thread::hardware_concurrency(), 1);
    http_threads = config.http_threads == 0 ? hw : config.http_threads;
    compute_threads = config.compute_threads == 0 ? hw : config.compute_threads;
    if (compute_threads > 1) {
      pool = std::make_unique<ThreadPool>(compute_threads - 1);
    }
    std::random_device rd;
    id_prefix = std::to_string((static_cast<std::uint64_t>(rd()) << 16U) ^ rd());
  }

  Catalog& catalog;
  ServerConfig config;
  std::size_t http_threads = 1;
  std::size_t compute_threads = 1;
  std::unique_ptr<ThreadPool> pool;
  std::counting_semaphore<> batch_slots{kBatchSlots};
  httplib::Server http;
  std::thread listener;
  std::atomic<int> port{0};
  std::atomic<bool> started{false};
  std::atomic<bool> accepting{false};
  const Clock::time_point created = Clock::now();
  std::string id_prefix;
  std::atomic<std::uint64_t> next_request{0};

  std::mutex flight_mutex;
  std::condition_variable flight_done;
  std::size_t in_flight = 0;

  using Handler = std::function<void(const httplib::Request&, httplib::Response&)>;

  [[nodiscard]] std::string request_id(const httplib::Request& req) {
    std::string id = req.get_header_value("X-Request-Id");
    if (is_valid_request_id(id)) {
      return id;
    }
    return id_prefix + "-" + std::to_string(next_request.fetch_add(1));
  }

  // Common handling for every route: request id, readiness, authentication, in-flight counting,
  // exceptions.
  [[nodiscard]] Handler wrap(Handler inner, bool open_route) {
    return [this, inner = std::move(inner), open_route](const httplib::Request& req,
                                                        httplib::Response& res) {
      res.set_header("X-Request-Id", request_id(req));
      if (!open_route) {
        if (!accepting.load()) {
          send_error(
              res,
              {.http_status = 503, .code = "UNAVAILABLE", .message = "server is shutting down"});
          return;
        }
        if (!config.api_key.empty() && !constant_time_equal(req.get_header_value("Authorization"),
                                                            "Bearer " + config.api_key)) {
          res.set_header("WWW-Authenticate", "Bearer");
          send_error(res, {.http_status = 401,
                           .code = "UNAUTHORIZED",
                           .message = "missing or wrong bearer token"});
          return;
        }
      }
      {
        const std::lock_guard<std::mutex> lock(flight_mutex);
        ++in_flight;
      }
      try {
        inner(req, res);
      } catch (const std::bad_alloc&) {
        send_error(res,
                   {.http_status = 507, .code = "RESOURCE_EXHAUSTED", .message = "out of memory"});
      } catch (const std::exception& e) {
        send_error(res, {.http_status = 500, .code = "INTERNAL", .message = e.what()});
      }
      const std::lock_guard<std::mutex> lock(flight_mutex);
      if (--in_flight == 0) {
        flight_done.notify_all();
      }
    };
  }

  // Requires a JSON content type; returns false after sending 415.
  static bool require_json(const httplib::Request& req, httplib::Response& res) {
    const std::string type = req.get_header_value("Content-Type");
    if (type.starts_with(kJson)) {
      return true;
    }
    send_error(res, {.http_status = 415,
                     .code = "UNSUPPORTED_MEDIA_TYPE",
                     .message = "Content-Type must be application/json"});
    return false;
  }

  [[nodiscard]] std::shared_ptr<Collection> find(const httplib::Request& req,
                                                 httplib::Response& res) const {
    Result<std::shared_ptr<Collection>> c = catalog.get(req.matches[1].str());
    if (!c.ok()) {
      send_status(res, c.status());
      return nullptr;
    }
    return std::move(c).value();
  }

  // Runs `body` with the compute pool (or inline when there is none or all slots are busy).
  void with_pool(const std::function<void(ThreadPool*)>& body) {
    if (pool == nullptr || !batch_slots.try_acquire()) {
      body(nullptr);
      return;
    }
    struct Release {
      explicit Release(std::counting_semaphore<>* s) noexcept : slots(s) {}
      std::counting_semaphore<>* slots;
      Release(const Release&) = delete;
      Release& operator=(const Release&) = delete;
      Release(Release&&) = delete;
      Release& operator=(Release&&) = delete;
      ~Release() { slots->release(); }
    } release{&batch_slots};
    body(pool.get());
  }

  void insert(const std::shared_ptr<Collection>& c, const InsertRequest& request,
              httplib::Response& res) {
    Result<std::size_t> inserted = Status::internal("not run");
    with_pool([&](ThreadPool* p) {
      inserted = c->add_batch(request.ids, request.rows, {.upsert = request.upsert}, p);
    });
    if (!inserted.ok()) {
      send_status(res, inserted.status());
      return;
    }
    send_json(res, 200, R"({"inserted": )" + std::to_string(inserted.value()) + "}");
  }

  void register_routes() {
    auto get = [this](std::string_view pattern, Handler h, bool open_route = false) {
      http.Get(std::string(pattern), wrap(std::move(h), open_route));
    };
    auto post = [this](std::string_view pattern, Handler h) {
      http.Post(std::string(pattern), wrap(std::move(h), false));
    };
    auto del = [this](std::string_view pattern, Handler h) {
      http.Delete(std::string(pattern), wrap(std::move(h), false));
    };
    const std::string coll = "/v1/collections/" + std::string(kName);

    get(
        "/healthz",
        [](const httplib::Request&, httplib::Response& res) {
          send_json(res, 200, R"({"status": "ok"})");
        },
        true);
    get(
        "/readyz",
        [this](const httplib::Request&, httplib::Response& res) {
          if (accepting.load()) {
            send_json(res, 200, R"({"status": "ready"})");
          } else {
            send_error(
                res, {.http_status = 503, .code = "UNAVAILABLE", .message = "server is not ready"});
          }
        },
        true);
    get("/v1/status", [this](const httplib::Request&, httplib::Response& res) {
      const auto uptime = std::chrono::duration<double>(Clock::now() - created).count();
      std::string body =
          R"({"version": ")" + std::string(kVersion) + R"(", "git_sha": ")" + std::string(kGitSha) +
          R"(", "build_type": ")" + std::string(kBuildType) + R"(", "compiler": ")" +
          std::string(kCompiler) + R"(", "simd": ")" + std::string(to_string(active_simd_level())) +
          R"(", "uptime_s": )" + format_ms(uptime) + R"(, "collections": )" +
          std::to_string(catalog.size()) + R"(, "http_threads": )" + std::to_string(http_threads) +
          R"(, "compute_threads": )" + std::to_string(compute_threads) + "}";
      send_json(res, 200, std::move(body));
    });

    post("/v1/collections", [this](const httplib::Request& req, httplib::Response& res) {
      if (!require_json(req, res)) {
        return;
      }
      Decoded<CreateRequest> request = decode_create(req.body, config.limits);
      if (!request.ok) {
        send_error(res, request.error);
        return;
      }
      Result<std::shared_ptr<Collection>> c =
          catalog.create(request.value.name, request.value.config);
      if (!c.ok()) {
        send_status(res, c.status());
        return;
      }
      send_json(res, 201, encode_collection(request.value.name, *c.value()));
    });
    get("/v1/collections", [this](const httplib::Request&, httplib::Response& res) {
      std::string body = R"({"collections": [)";
      bool first = true;
      for (const std::string& name : catalog.list()) {
        Result<std::shared_ptr<Collection>> c = catalog.get(name);
        if (!c.ok()) {
          continue;  // dropped meanwhile
        }
        body += (first ? "" : ", ") + encode_collection(name, *c.value());
        first = false;
      }
      send_json(res, 200, body + "]}");
    });
    get(coll, [this](const httplib::Request& req, httplib::Response& res) {
      if (const auto c = find(req, res)) {
        send_json(res, 200, encode_collection(req.matches[1].str(), *c));
      }
    });
    del(coll, [this](const httplib::Request& req, httplib::Response& res) {
      if (const Status st = catalog.drop(req.matches[1].str()); !st.ok()) {
        send_status(res, st);
        return;
      }
      send_json(res, 200, R"({"dropped": ")" + req.matches[1].str() + R"("})");
    });

    post(coll + "/vectors", [this](const httplib::Request& req, httplib::Response& res) {
      if (!require_json(req, res)) {
        return;
      }
      const auto c = find(req, res);
      if (!c) {
        return;
      }
      Decoded<InsertRequest> request = decode_insert(req.body, c->config().dim, config.limits);
      if (!request.ok) {
        send_error(res, request.error);
        return;
      }
      insert(c, request.value, res);
    });
    post(coll + "/vectors:bulk", [this](const httplib::Request& req, httplib::Response& res) {
      if (!req.get_header_value("Content-Type").starts_with("application/octet-stream")) {
        send_error(res, {.http_status = 415,
                         .code = "UNSUPPORTED_MEDIA_TYPE",
                         .message = "Content-Type must be application/octet-stream"});
        return;
      }
      const auto c = find(req, res);
      if (!c) {
        return;
      }
      Decoded<InsertRequest> request = decode_bulk(req.body, c->config().dim, config.limits);
      if (!request.ok) {
        send_error(res, request.error);
        return;
      }
      const std::string upsert = req.get_param_value("upsert");
      if (!upsert.empty() && upsert != "true" && upsert != "false") {
        send_error(res, {.http_status = 400,
                         .code = "INVALID_ARGUMENT",
                         .message = "query parameter 'upsert' must be true or false"});
        return;
      }
      request.value.upsert = upsert == "true";
      insert(c, request.value, res);
    });
    get(coll + R"(/vectors/(\d{1,20}))",
        [this](const httplib::Request& req, httplib::Response& res) {
          const auto c = find(req, res);
          ExternalId id = 0;
          if (!c) {
            return;
          }
          if (!parse_id(req.matches[2].str(), id)) {
            send_status(res, Status::invalid_argument("invalid vector id"));
            return;
          }
          Result<std::vector<float>> v = c->get(id);
          if (!v.ok()) {
            send_status(res, v.status());
            return;
          }
          send_json(res, 200, encode_vector(id, v.value()));
        });
    del(coll + R"(/vectors/(\d{1,20}))",
        [this](const httplib::Request& req, httplib::Response& res) {
          const auto c = find(req, res);
          ExternalId id = 0;
          if (!c) {
            return;
          }
          if (!parse_id(req.matches[2].str(), id)) {
            send_status(res, Status::invalid_argument("invalid vector id"));
            return;
          }
          if (const Status st = c->remove(id); !st.ok()) {
            send_status(res, st);
            return;
          }
          send_json(res, 200, R"({"deleted": )" + std::to_string(id) + "}");
        });

    post(coll + "/search", [this](const httplib::Request& req, httplib::Response& res) {
      const Clock::time_point start = Clock::now();
      if (!require_json(req, res)) {
        return;
      }
      const auto c = find(req, res);
      if (!c) {
        return;
      }
      Decoded<SearchRequest> request =
          decode_search(req.body, c->config().dim, false, config.limits);
      if (!request.ok) {
        send_error(res, request.error);
        return;
      }
      Result<std::vector<Neighbor>> hits = c->search(request.value.vectors, request.value.params);
      if (!hits.ok()) {
        send_status(res, hits.status());
        return;
      }
      send_json(res, 200,
                R"({"results": )" + encode_neighbors(hits.value()) + R"(, "took_ms": )" +
                    format_ms(ms_since(start)) + "}");
    });
    post(coll + "/search:batch", [this](const httplib::Request& req, httplib::Response& res) {
      const Clock::time_point start = Clock::now();
      if (!require_json(req, res)) {
        return;
      }
      const auto c = find(req, res);
      if (!c) {
        return;
      }
      Decoded<SearchRequest> request =
          decode_search(req.body, c->config().dim, true, config.limits);
      if (!request.ok) {
        send_error(res, request.error);
        return;
      }
      const SearchRequest& q = request.value;
      const std::size_t k = q.params.k;
      std::vector<ExternalId> ids(q.count * k);
      std::vector<float> distances(q.count * k);
      std::vector<std::uint32_t> counts(q.count);
      Status st;
      with_pool([&](ThreadPool* p) {
        st = c->search_batch(q.vectors, q.count, q.params, ids, distances, counts, p);
      });
      if (!st.ok()) {
        send_status(res, st);
        return;
      }
      std::string body = R"({"results": [)";
      std::vector<Neighbor> row;
      for (std::size_t i = 0; i < q.count; ++i) {
        row.clear();
        for (std::size_t j = 0; j < counts[i]; ++j) {
          row.push_back({.id = ids[(i * k) + j], .distance = distances[(i * k) + j]});
        }
        body += (i == 0 ? "" : ", ") + encode_neighbors(row);
      }
      send_json(res, 200, body + R"(], "took_ms": )" + format_ms(ms_since(start)) + "}");
    });

    get(coll + "/stats", [this](const httplib::Request& req, httplib::Response& res) {
      if (const auto c = find(req, res)) {
        send_json(res, 200, encode_stats(req.matches[1].str(), *c));
      }
    });
    post(coll + "/snapshot", [this](const httplib::Request& req, httplib::Response& res) {
      const Clock::time_point start = Clock::now();
      Result<SnapshotInfo> info = catalog.snapshot(req.matches[1].str());
      if (!info.ok()) {
        send_status(res, info.status());
        return;
      }
      send_json(res, 200,
                R"({"generation": )" + std::to_string(info.value().generation) + R"(, "bytes": )" +
                    std::to_string(info.value().bytes) + R"(, "took_ms": )" +
                    format_ms(ms_since(start)) + "}");
    });
    post(coll + "/compact", [this](const httplib::Request& req, httplib::Response& res) {
      const Clock::time_point start = Clock::now();
      const auto c = find(req, res);
      if (!c) {
        return;
      }
      Result<CompactStats> stats = c->compact();
      if (!stats.ok()) {
        send_status(res, stats.status());
        return;
      }
      send_json(res, 200,
                R"({"rows_before": )" + std::to_string(stats.value().rows_before) +
                    R"(, "removed_rows": )" + std::to_string(stats.value().removed_rows) +
                    R"(, "rows_after": )" + std::to_string(stats.value().rows_after) +
                    R"(, "took_ms": )" + format_ms(ms_since(start)) + "}");
    });
  }

  void configure() {
    const std::size_t threads = http_threads;
    http.new_task_queue = [threads] { return new httplib::ThreadPool(threads); };
    http.set_payload_max_length(config.limits.max_body_bytes);
    http.set_read_timeout(config.read_timeout);
    http.set_write_timeout(config.write_timeout);
    http.set_keep_alive_timeout(config.keep_alive_timeout);
    // cpp-httplib's default (SO_REUSEPORT on POSIX, SO_REUSEADDR on Windows) lets a second process
    // bind the same port and receive part of the traffic. Only allow restarts over TIME_WAIT
    // sockets on POSIX, and claim the port exclusively on Windows.
    http.set_socket_options([](auto sock) {
#if defined(_WIN32)
      static_cast<void>(httplib::set_socket_opt(sock, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, 1));
#else
      static_cast<void>(httplib::set_socket_opt(sock, SOL_SOCKET, SO_REUSEADDR, 1));
#endif
    });
    // Errors produced by httplib itself (unknown route, body too large, ...) get a JSON body.
    http.set_error_handler([this](const httplib::Request& req, httplib::Response& res) {
      if (!res.body.empty()) {
        return;  // a handler already produced an error body
      }
      res.set_header("X-Request-Id", request_id(req));
      ApiError error{.http_status = res.status, .code = "INVALID_ARGUMENT", .message = ""};
      switch (res.status) {
        case 404:
          error.code = "NOT_FOUND";
          error.message = "no route for " + req.method + " " + req.path;
          break;
        case 405:
          error.code = "METHOD_NOT_ALLOWED";
          error.message = "method not allowed";
          break;
        case 413:
          error.code = "PAYLOAD_TOO_LARGE";
          error.message =
              "request body exceeds " + std::to_string(config.limits.max_body_bytes) + " bytes";
          break;
        default:
          error.message = "request rejected";
          break;
      }
      send_error(res, error);
    });
    http.set_exception_handler(
        [](const httplib::Request&, httplib::Response& res, const std::exception_ptr&) {
          send_error(res, {.http_status = 500, .code = "INTERNAL", .message = "internal error"});
        });
    register_routes();
  }
};

Server::Server(Catalog& catalog, ServerConfig config)
    : impl_(std::make_unique<Impl>(catalog, std::move(config))) {
  impl_->configure();
}

Server::~Server() {
  stop();
}

Status Server::start() {
  if (impl_->started.exchange(true)) {
    return {ErrorCode::FailedPrecondition, "server already started"};
  }
  Impl& s = *impl_;
  int port = s.config.port;
  if (port == 0) {
    port = s.http.bind_to_any_port(s.config.host);
    if (port <= 0) {
      return Status::io_error("cannot bind " + s.config.host + " to a free port");
    }
  } else if (!s.http.bind_to_port(s.config.host, port)) {
    return Status::io_error("cannot bind " + s.config.host + ":" + std::to_string(port));
  }
  s.port.store(port);
  s.accepting.store(true);
  s.listener = std::thread([&s] { static_cast<void>(s.http.listen_after_bind()); });
  s.http.wait_until_ready();
  return {};
}

void Server::stop() {
  Impl& s = *impl_;
  if (!s.listener.joinable()) {
    return;
  }
  s.accepting.store(false);
  {
    std::unique_lock<std::mutex> lock(s.flight_mutex);
    s.flight_done.wait_for(lock, s.config.drain_timeout, [&s] { return s.in_flight == 0; });
  }
  s.http.stop();
  s.listener.join();
}

int Server::port() const noexcept {
  return impl_->port.load();
}

bool Server::ready() const noexcept {
  return impl_->accepting.load();
}

std::span<const RouteInfo> Server::routes() noexcept {
  return kRoutes;
}

}  // namespace vf::server
