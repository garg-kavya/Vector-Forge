#pragma once

// Prometheus metrics of the HTTP server (docs/http-api.md, "Metrics").
//
// Per route: a request counter by status class and a latency histogram with fixed, roughly
// logarithmic buckets (50 µs … 10 s). All counters are std::atomic, so observe() never locks;
// render() reads them without stopping writers (a scrape may see a request in the counter but not
// yet in the histogram). Collection gauges are read from the catalog at scrape time.

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <vectorforge/catalog.hpp>

namespace vf::server {

struct RouteInfo;

inline constexpr std::array<double, 17> kLatencyBucketsSeconds{
    0.00005, 0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025,
    0.05,    0.1,    0.25,    0.5,    1.0,   2.5,    5.0,   10.0};

class LatencyHistogram {
 public:
  void observe(double seconds) noexcept;
  // Cumulative bucket counts (Prometheus `le` semantics), then +Inf.
  [[nodiscard]] std::array<std::uint64_t, kLatencyBucketsSeconds.size() + 1> cumulative()
      const noexcept;
  [[nodiscard]] std::uint64_t count() const noexcept { return count_.load(); }
  [[nodiscard]] double sum_seconds() const noexcept;

 private:
  std::array<std::atomic<std::uint64_t>, kLatencyBucketsSeconds.size() + 1> buckets_{};
  std::atomic<std::uint64_t> count_{0};
  std::atomic<std::uint64_t> sum_ns_{0};
};

class Metrics {
 public:
  // Routes are the server's route table; index routes.size() stands for "unmatched".
  explicit Metrics(std::span<const RouteInfo> routes);

  [[nodiscard]] std::size_t unmatched_route() const noexcept { return routes_.size(); }
  void observe(std::size_t route, int http_status, double seconds) noexcept;
  void set_in_flight(std::size_t value) noexcept { in_flight_.store(value); }

  // The text exposition format, version 0.0.4.
  [[nodiscard]] std::string render(const Catalog& catalog, double uptime_seconds) const;

 private:
  struct PerRoute {
    std::array<std::atomic<std::uint64_t>, 5> by_class{};  // 1xx … 5xx
    LatencyHistogram latency;
  };

  std::span<const RouteInfo> routes_;
  // Atomics are not movable, so no std::vector.
  std::unique_ptr<PerRoute[]> per_route_;  // NOLINT(modernize-avoid-c-arrays)
  std::atomic<std::size_t> in_flight_{0};
};

// Escapes a Prometheus label value (backslash, double quote, newline).
[[nodiscard]] std::string escape_label(std::string_view value);

}  // namespace vf::server
