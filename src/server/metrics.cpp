#include "server/metrics.hpp"

#include <algorithm>
#include <cstdio>
#include <string_view>

#include <vectorforge/simd.hpp>
#include <vectorforge/version.hpp>

#include "server/server.hpp"

namespace vf::server {

namespace {

std::string number(double v) {
  std::array<char, 32> buf{};
  const int n = std::snprintf(buf.data(), buf.size(), "%.9g", v);
  return {buf.data(), static_cast<std::size_t>(n > 0 ? n : 0)};
}

void header(std::string& out, std::string_view name, std::string_view type, std::string_view help) {
  out += "# HELP ";
  out += name;
  out += ' ';
  out += help;
  out += "\n# TYPE ";
  out += name;
  out += ' ';
  out += type;
  out += '\n';
}

}  // namespace

std::string escape_label(std::string_view value) {
  std::string out;
  out.reserve(value.size());
  for (const char c : value) {
    if (c == '\\') {
      out += "\\\\";
    } else if (c == '"') {
      out += "\\\"";
    } else if (c == '\n') {
      out += "\\n";
    } else {
      out += c;
    }
  }
  return out;
}

void LatencyHistogram::observe(double seconds) noexcept {
  const auto it =
      std::lower_bound(kLatencyBucketsSeconds.begin(), kLatencyBucketsSeconds.end(), seconds);
  const auto index = static_cast<std::size_t>(it - kLatencyBucketsSeconds.begin());
  buckets_[index].fetch_add(1, std::memory_order_relaxed);
  count_.fetch_add(1, std::memory_order_relaxed);
  const double ns = std::max(seconds, 0.0) * 1e9;
  sum_ns_.fetch_add(static_cast<std::uint64_t>(ns), std::memory_order_relaxed);
}

std::array<std::uint64_t, kLatencyBucketsSeconds.size() + 1> LatencyHistogram::cumulative()
    const noexcept {
  std::array<std::uint64_t, kLatencyBucketsSeconds.size() + 1> out{};
  std::uint64_t running = 0;
  for (std::size_t i = 0; i < out.size(); ++i) {
    running += buckets_[i].load(std::memory_order_relaxed);
    out[i] = running;
  }
  return out;
}

double LatencyHistogram::sum_seconds() const noexcept {
  return static_cast<double>(sum_ns_.load(std::memory_order_relaxed)) / 1e9;
}

Metrics::Metrics(std::span<const RouteInfo> routes)
    : routes_(routes),
      per_route_(
          std::make_unique<PerRoute[]>(routes.size() + 1)) {  // NOLINT(modernize-avoid-c-arrays)
}

void Metrics::observe(std::size_t route, int http_status, double seconds) noexcept {
  PerRoute& r = per_route_[std::min(route, routes_.size())];
  const int status_class = std::clamp(http_status / 100, 1, 5);
  r.by_class[static_cast<std::size_t>(status_class - 1)].fetch_add(1, std::memory_order_relaxed);
  r.latency.observe(seconds);
}

std::string Metrics::render(const Catalog& catalog, double uptime_seconds) const {
  std::string out;
  out.reserve(8192);
  auto labels = [this](std::size_t i) {
    if (i == routes_.size()) {
      return std::string(R"(method="",route="unmatched")");
    }
    return "method=\"" + escape_label(routes_[i].method) + "\",route=\"" +
           escape_label(routes_[i].path) + "\"";
  };

  header(out, "vectorforge_http_requests_total", "counter",
         "HTTP requests by route and status class.");
  for (std::size_t i = 0; i <= routes_.size(); ++i) {
    for (std::size_t c = 0; c < 5; ++c) {
      const std::uint64_t v = per_route_[i].by_class[c].load(std::memory_order_relaxed);
      if (v == 0) {
        continue;
      }
      out += "vectorforge_http_requests_total{" + labels(i) + ",status=\"" + std::to_string(c + 1) +
             "xx\"} " + std::to_string(v) + "\n";
    }
  }

  header(out, "vectorforge_http_request_duration_seconds", "histogram",
         "Time from routing to response, by route.");
  for (std::size_t i = 0; i <= routes_.size(); ++i) {
    const LatencyHistogram& h = per_route_[i].latency;
    if (h.count() == 0) {
      continue;
    }
    const auto cumulative = h.cumulative();
    const std::string l = labels(i);
    for (std::size_t b = 0; b < kLatencyBucketsSeconds.size(); ++b) {
      out += "vectorforge_http_request_duration_seconds_bucket{" + l + ",le=\"" +
             number(kLatencyBucketsSeconds[b]) + "\"} " + std::to_string(cumulative[b]) + "\n";
    }
    out += "vectorforge_http_request_duration_seconds_bucket{" + l + ",le=\"+Inf\"} " +
           std::to_string(cumulative.back()) + "\n";
    out += "vectorforge_http_request_duration_seconds_sum{" + l + "} " + number(h.sum_seconds()) +
           "\n";
    out += "vectorforge_http_request_duration_seconds_count{" + l + "} " +
           std::to_string(cumulative.back()) + "\n";
  }

  header(out, "vectorforge_http_in_flight_requests", "gauge", "Requests being handled.");
  out += "vectorforge_http_in_flight_requests " + std::to_string(in_flight_.load()) + "\n";

  const std::vector<CatalogEntry> entries = catalog.entries();
  header(out, "vectorforge_collections", "gauge", "Collections in the catalog.");
  out += "vectorforge_collections " + std::to_string(entries.size()) + "\n";
  header(out, "vectorforge_collection_vectors", "gauge", "Live vectors per collection.");
  for (const CatalogEntry& e : entries) {
    out += "vectorforge_collection_vectors{collection=\"" + escape_label(e.name) + "\"} " +
           std::to_string(e.size) + "\n";
  }
  header(out, "vectorforge_collection_deleted_vectors", "gauge",
         "Removed vectors not yet compacted, per collection.");
  for (const CatalogEntry& e : entries) {
    out += "vectorforge_collection_deleted_vectors{collection=\"" + escape_label(e.name) + "\"} " +
           std::to_string(e.deleted) + "\n";
  }
  header(out, "vectorforge_collection_memory_bytes", "gauge",
         "Heap memory per collection (vectors, ids, tombstones, index; mapped files excluded).");
  for (const CatalogEntry& e : entries) {
    out += "vectorforge_collection_memory_bytes{collection=\"" + escape_label(e.name) + "\"} " +
           std::to_string(e.memory_bytes) + "\n";
  }
  header(out, "vectorforge_collection_snapshot_generation", "gauge",
         "Last snapshot generation per collection (0: never snapshotted).");
  for (const CatalogEntry& e : entries) {
    out += "vectorforge_collection_snapshot_generation{collection=\"" + escape_label(e.name) +
           "\"} " + std::to_string(e.last_generation) + "\n";
  }

  header(out, "vectorforge_uptime_seconds", "gauge", "Seconds since the server was created.");
  out += "vectorforge_uptime_seconds " + number(uptime_seconds) + "\n";
  header(out, "vectorforge_build_info", "gauge", "Build information (always 1).");
  out += "vectorforge_build_info{version=\"" + escape_label(kVersion) + "\",git_sha=\"" +
         escape_label(kGitSha) + "\",simd=\"" + std::string(to_string(active_simd_level())) +
         "\"} 1\n";
  return out;
}

}  // namespace vf::server
