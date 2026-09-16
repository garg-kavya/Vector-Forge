// HTTP overhead benchmark (docs/DESIGN.md §21 Phase 7): single-query latency in-process vs over
// HTTP on loopback (with the request decoding share), and bulk ingestion throughput through the
// JSON and binary insert endpoints.
//
//   vf_http_bench --n 100000 --dim 128 --queries 2000 --out http.json

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <httplib.h>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <vectorforge/catalog.hpp>
#include <vectorforge/simd.hpp>
#include <vectorforge/thread_pool.hpp>
#include <vectorforge/version.hpp>

#include "server/json_codec.hpp"
#include "server/server.hpp"
#include "simd/cpu_features.hpp"
#include "util/synthetic.hpp"
#include "util/timer.hpp"

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

namespace {

namespace d = vf::detail;
using json = nlohmann::json;

struct Options {
  std::uint64_t n = 100000;
  std::uint32_t dim = 128;
  std::uint64_t queries = 2000;
  std::uint32_t k = 10;
  std::uint32_t ef = 64;
  std::uint64_t ingest = 100000;
  std::uint32_t batch = 1000;
  std::string out;
};

std::vector<float> generate(const Options& o, std::uint64_t rows, std::uint64_t seed) {
  d::SyntheticSpec spec;
  spec.distribution = d::SyntheticDistribution::GaussianMixture;
  spec.dim = o.dim;
  spec.clusters = 100;
  spec.spread = 0.1F;
  spec.seed = seed;
  spec.mixture_seed = 7936;
  d::SyntheticGenerator gen = d::SyntheticGenerator::create(spec).value();
  std::vector<float> data(static_cast<std::size_t>(rows) * o.dim);
  for (std::uint64_t r = 0; r < rows; ++r) {
    gen.next_row(std::span<float>(data).subspan(static_cast<std::size_t>(r) * o.dim, o.dim));
  }
  return data;
}

// Temporary catalog directory, removed on exit.
class TempDir {
 public:
  TempDir() {
    std::random_device rd;
    path_ = std::filesystem::temp_directory_path() / ("vf_http_bench_" + std::to_string(rd()));
    std::filesystem::create_directories(path_);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
  TempDir(TempDir&&) = delete;
  TempDir& operator=(TempDir&&) = delete;
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }
  [[nodiscard]] const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

std::string summary(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  auto pct = [&samples](double p) {
    const auto rank =
        static_cast<std::size_t>(std::ceil(p / 100.0 * static_cast<double>(samples.size())));
    return samples[std::min(samples.size() - 1, rank == 0 ? 0 : rank - 1)] * 1e6;
  };
  std::ostringstream out;
  double sum = 0;
  for (const double s : samples) {
    sum += s;
  }
  out << "{\"mean_us\": " << sum / static_cast<double>(samples.size()) * 1e6
      << ", \"p50_us\": " << pct(50) << ", \"p99_us\": " << pct(99) << "}";
  return out.str();
}

std::string bulk_body(std::uint32_t dim, std::span<const vf::ExternalId> ids,
                      std::span<const float> rows) {
  std::string body(16 + (ids.size() * 8) + (rows.size() * 4), '\0');
  std::memcpy(body.data(), "VFB1", 4);
  const std::uint64_t count = ids.size();
  std::memcpy(body.data() + 4, &dim, 4);  // little-endian hosts only (x86-64)
  std::memcpy(body.data() + 8, &count, 8);
  std::memcpy(body.data() + 16, ids.data(), ids.size() * 8);
  std::memcpy(body.data() + 16 + (ids.size() * 8), rows.data(), rows.size() * 4);
  return body;
}

int run(const Options& o) {
  const TempDir dir;
  auto catalog = vf::Catalog::open(dir.path()).value();
  vf::CollectionConfig cfg;
  cfg.dim = o.dim;
  auto hnsw = catalog->create("hnsw", cfg).value();
  const std::vector<float> base = generate(o, o.n, 1);
  const std::vector<float> queries = generate(o, o.queries, 2);
  std::vector<vf::ExternalId> ids(static_cast<std::size_t>(std::max(o.n, o.ingest)));
  for (std::size_t i = 0; i < ids.size(); ++i) {
    ids[i] = i;
  }
  {
    vf::ThreadPool pool(std::max<unsigned>(std::thread::hardware_concurrency(), 2U) - 1);
    std::cerr << "building " << o.n << " vectors\n";
    static_cast<void>(
        hnsw->add_batch(std::span<const vf::ExternalId>(ids).first(o.n), base, {}, &pool));
  }

  vf::server::ServerConfig scfg;
  scfg.port = 0;
  scfg.limits.max_body_bytes = std::size_t{512} << 20U;
  scfg.limits.max_batch = std::max<std::size_t>(o.batch, 10000);
  vf::server::Server server(*catalog, scfg);
  if (!server.start().ok()) {
    std::cerr << "cannot start server\n";
    return 1;
  }
  httplib::Client client("127.0.0.1", server.port());
  client.set_keep_alive(true);

  vf::SearchParams params;
  params.k = o.k;
  params.ef_search = o.ef;
  const auto nq = static_cast<std::size_t>(o.queries);
  std::vector<std::string> bodies(nq);
  for (std::size_t q = 0; q < nq; ++q) {
    const auto v = std::span<const float>(queries).subspan(q * o.dim, o.dim);
    bodies[q] =
        json({{"vector", std::vector<float>(v.begin(), v.end())}, {"k", o.k}, {"ef_search", o.ef}})
            .dump();
  }
  // Warm-up (connection, contexts, caches).
  for (std::size_t q = 0; q < std::min<std::size_t>(nq, 200); ++q) {
    static_cast<void>(
        hnsw->search(std::span<const float>(queries).subspan(q * o.dim, o.dim), params));
    static_cast<void>(client.Post("/v1/collections/hnsw/search", bodies[q], "application/json"));
  }
  std::vector<double> in_process;
  std::vector<double> over_http;
  std::vector<double> decode;
  std::vector<double> server_side;
  std::size_t body_bytes = 0;
  std::size_t response_bytes = 0;
  for (std::size_t q = 0; q < nq; ++q) {
    const auto v = std::span<const float>(queries).subspan(q * o.dim, o.dim);
    {
      const d::Stopwatch sw;
      static_cast<void>(hnsw->search(v, params));
      in_process.push_back(sw.elapsed_seconds());
    }
    {
      const d::Stopwatch sw;
      static_cast<void>(vf::server::decode_search(bodies[q], o.dim, false, scfg.limits));
      decode.push_back(sw.elapsed_seconds());
    }
    const d::Stopwatch sw;
    auto res = client.Post("/v1/collections/hnsw/search", bodies[q], "application/json");
    over_http.push_back(sw.elapsed_seconds());
    if (!res || res->status != 200) {
      std::cerr << "search failed\n";
      return 1;
    }
    server_side.push_back(json::parse(res->body)["took_ms"].get<double>() / 1000.0);
    body_bytes += bodies[q].size();
    response_bytes += res->body.size();
  }

  // Ingestion into Flat collections, so that transfer and decoding dominate.
  vf::CollectionConfig flat_cfg;
  flat_cfg.dim = o.dim;
  flat_cfg.index = vf::IndexType::Flat;
  static_cast<void>(catalog->create("ingest_json", flat_cfg));
  static_cast<void>(catalog->create("ingest_bulk", flat_cfg));
  const std::vector<float> rows = generate(o, o.ingest, 3);
  const auto total = static_cast<std::size_t>(o.ingest);
  double json_seconds = 0;
  double json_encode_seconds = 0;
  double bulk_seconds = 0;
  std::size_t json_bytes = 0;
  std::size_t bulk_bytes = 0;
  for (std::size_t first = 0; first < total; first += o.batch) {
    const std::size_t count = std::min<std::size_t>(o.batch, total - first);
    const auto id_span = std::span<const vf::ExternalId>(ids).subspan(first, count);
    const auto row_span = std::span<const float>(rows).subspan(first * o.dim, count * o.dim);
    d::Stopwatch encode_watch;
    json vectors = json::array();
    for (std::size_t i = 0; i < count; ++i) {
      const auto r = row_span.subspan(i * o.dim, o.dim);
      vectors.push_back({{"id", id_span[i]}, {"vector", std::vector<float>(r.begin(), r.end())}});
    }
    const std::string body = json({{"vectors", vectors}}).dump();
    json_encode_seconds += encode_watch.elapsed_seconds();
    json_bytes += body.size();
    d::Stopwatch json_watch;
    auto a = client.Post("/v1/collections/ingest_json/vectors", body, "application/json");
    json_seconds += json_watch.elapsed_seconds();
    const std::string binary = bulk_body(o.dim, id_span, row_span);
    bulk_bytes += binary.size();
    d::Stopwatch bulk_watch;
    auto b =
        client.Post("/v1/collections/ingest_bulk/vectors:bulk", binary, "application/octet-stream");
    bulk_seconds += bulk_watch.elapsed_seconds();
    if (!a || a->status != 200 || !b || b->status != 200) {
      std::cerr << "ingest failed\n";
      return 1;
    }
  }
  server.stop();

  std::ostringstream out;
  out << "{\n  \"schema\": 1, \"suite\": \"http\",\n"
      << "  \"git\": {\"sha\": \"" << vf::kGitSha << "\"},\n"
      << "  \"machine\": {\"cpu\": \"" << d::cpu_features().brand << "\"},\n"
      << "  \"build\": {\"compiler\": \"" << vf::kCompiler << "\", \"type\": \"" << vf::kBuildType
      << "\", \"simd_tier\": \"" << vf::to_string(vf::active_simd_level()) << "\"},\n"
      << "  \"params\": {\"n\": " << o.n << ", \"dim\": " << o.dim << ", \"index\": \"hnsw\""
      << ", \"k\": " << o.k << ", \"ef_search\": " << o.ef << ", \"queries\": " << nq
      << ", \"ingest\": " << o.ingest << ", \"batch\": " << o.batch
      << ", \"transport\": \"loopback, keep-alive, one client thread\"},\n"
      << "  \"search\": {\"in_process\": " << summary(in_process)
      << ", \"http_round_trip\": " << summary(over_http)
      << ", \"server_handling\": " << summary(server_side)
      << ", \"request_decode\": " << summary(decode)
      << ", \"request_bytes_mean\": " << body_bytes / nq
      << ", \"response_bytes_mean\": " << response_bytes / nq << "},\n"
      << "  \"ingest\": {\"json_vectors_per_s\": " << static_cast<double>(total) / json_seconds
      << ", \"json_client_encode_s\": " << json_encode_seconds << ", \"json_bytes\": " << json_bytes
      << ", \"bulk_vectors_per_s\": " << static_cast<double>(total) / bulk_seconds
      << ", \"bulk_bytes\": " << bulk_bytes << "}\n}\n";
  std::cout << out.str();
  if (!o.out.empty()) {
    std::ofstream(o.out, std::ios::binary) << out.str();
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Options o;
  CLI::App app{"VectorForge HTTP overhead benchmark"};
  app.add_option("--n", o.n, "indexed vectors");
  app.add_option("--dim", o.dim, "dimension");
  app.add_option("--queries", o.queries, "timed single queries");
  app.add_option("--k", o.k, "neighbours");
  app.add_option("--ef-search", o.ef, "HNSW beam");
  app.add_option("--ingest", o.ingest, "vectors inserted through each insert endpoint");
  app.add_option("--batch", o.batch, "vectors per insert request");
  app.add_option("--out", o.out, "JSON output file");
  CLI11_PARSE(app, argc, argv);
  return run(o);
}
