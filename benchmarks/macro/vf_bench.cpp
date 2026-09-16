// vf_bench (minimal, Phases 3-6): build an HNSW index over a seeded synthetic dataset (or load one
// with --index-file), sweep ef_search and report recall@k, latency, QPS and distance computations
// per query against exact (Flat) ground truth, plus Flat latency on the same queries. One
// configuration per process (docs/DESIGN.md §16). Single thread. --simd selects the kernel tier
// through VF_SIMD for A/B runs of the same binary on the same index file; --prefetch on|off sets
// HnswSearchOptions::prefetch (default on, the library default).
//
// The index is built through HnswBackend directly (the same code path Collection uses, minus id
// mapping) so that graph statistics and distance counters are observable.

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <vectorforge/collection.hpp>
#include <vectorforge/simd.hpp>
#include <vectorforge/thread_pool.hpp>
#include <vectorforge/version.hpp>

#include "collection/collection_factory.hpp"
#include "collection/collection_state.hpp"
#include "core/vector_ops.hpp"
#include "index/hnsw/hnsw_backend.hpp"
#include "index/hnsw/hnsw_validator.hpp"
#include "process_memory.hpp"
#include "simd/cpu_features.hpp"
#include "simd/kernels.hpp"
#include "storage/tombstones.hpp"
#include "storage/vector_store.hpp"
#include "util/dataset_io.hpp"
#include "util/recall.hpp"
#include "util/synthetic.hpp"
#include "util/timer.hpp"

#include <CLI/CLI.hpp>

namespace {

using vf::ExternalId;
using vf::Neighbor;
namespace d = vf::detail;

struct Options {
  std::uint64_t n = 100000;
  std::uint64_t queries = 1000;
  std::uint32_t dim = 128;
  std::string metric = "l2";
  std::string distribution = "gmm";
  std::uint32_t clusters = 100;
  float spread = 0.1F;
  std::uint64_t seed = 1;
  std::uint32_t m = 16;
  std::uint32_t ef_construction = 200;
  std::vector<std::uint32_t> ef_search{10, 16, 32, 64, 128, 256, 512};
  std::uint32_t k = 10;
  std::string selection = "heuristic";
  bool no_repair = false;
  bool skip_flat_timing = false;
  std::string out;
  // Storage scenarios (docs/DESIGN.md §16.3 "storage"): run each in a fresh process.
  std::string scenario = "sweep";  // sweep | save | load
  std::string index_file;
  std::string index_type = "hnsw";  // save: flat | hnsw
  std::string load_mode = "mmap";   // load: heap | mmap
  bool prefault = false;
  std::string simd;             // "" (leave VF_SIMD alone) | auto | scalar | avx2
  std::string prefetch = "on";  // sweep: on | off
  // threads / ingest scenarios
  std::vector<std::uint32_t> threads{1, 2, 4, 8, 16};
  std::uint32_t repeat = 5;
  std::uint32_t batch = 1000;
  // ingest / build: HNSW insert synchronisation, and threads used by the ingest writer
  std::string concurrency = "concurrent";
  std::uint32_t writer_threads = 1;
  std::string queries_file;  // threads: .npy/.fvecs queries instead of generated ones
};

vf::Concurrency concurrency_of(const Options& o) {
  return vf::parse_concurrency(o.concurrency).value();
}

void set_simd_request(const std::string& request) {
#if defined(_WIN32)
  _putenv_s("VF_SIMD", request.c_str());
#else
  setenv("VF_SIMD", request.c_str(), 1);  // NOLINT(concurrency-mt-unsafe): single-threaded startup
#endif
}

std::vector<float> generate(const Options& o, std::uint64_t rows, std::uint64_t seed) {
  d::SyntheticSpec spec;
  spec.distribution = o.distribution == "uniform" ? d::SyntheticDistribution::Uniform
                                                  : d::SyntheticDistribution::GaussianMixture;
  spec.dim = o.dim;
  spec.clusters = o.clusters;
  spec.spread = o.spread;
  spec.seed = seed;
  spec.mixture_seed = o.seed * 7919U + 17U;
  d::SyntheticGenerator gen = d::SyntheticGenerator::create(spec).value();
  std::vector<float> data(static_cast<std::size_t>(rows) * o.dim);
  for (std::uint64_t r = 0; r < rows; ++r) {
    gen.next_row(std::span<float>(data).subspan(static_cast<std::size_t>(r) * o.dim, o.dim));
  }
  return data;
}

double percentile(std::vector<double> sorted_samples, double p) {
  if (sorted_samples.empty()) {
    return 0.0;
  }
  // Nearest-rank definition on sorted samples.
  const auto rank =
      static_cast<std::size_t>(std::ceil(p / 100.0 * static_cast<double>(sorted_samples.size())));
  return sorted_samples[std::min(sorted_samples.size() - 1, rank == 0 ? 0 : rank - 1)];
}

void emit(const Options& o, const std::string& json) {
  std::cout << json;
  if (!o.out.empty()) {
    std::ofstream file(o.out, std::ios::binary);
    file << json;
  }
}

std::string environment_json() {
  std::ostringstream json;
  json << "  \"git\": {\"sha\": \"" << vf::kGitSha << "\"},\n"
       << "  \"machine\": {\"cpu\": \"" << d::cpu_features().brand << "\"},\n"
       << "  \"build\": {\"compiler\": \"" << vf::kCompiler << "\", \"type\": \"" << vf::kBuildType
       << "\", \"simd_tier\": \"" << vf::to_string(vf::active_simd_level())
       << "\", \"kernel_table\": \"" << d::kernels().name << "\"},\n";
  return json.str();
}

// save: build a collection through the public API and time Collection::save.
int run_save(const Options& o, vf::Metric metric) {
  if (o.index_file.empty()) {
    std::cerr << "--index-file is required\n";
    return 2;
  }
  std::vector<float> base = generate(o, o.n, o.seed);
  vf::CollectionConfig cfg;
  cfg.dim = o.dim;
  cfg.metric = metric;
  cfg.index = o.index_type == "flat" ? vf::IndexType::Flat : vf::IndexType::Hnsw;
  cfg.hnsw.M = o.m;
  cfg.hnsw.ef_construction = o.ef_construction;
  const std::unique_ptr<vf::Collection> c = vf::Collection::create(cfg).value();
  std::vector<ExternalId> ids(static_cast<std::size_t>(o.n));
  for (std::size_t i = 0; i < ids.size(); ++i) {
    ids[i] = i;
  }
  const d::Stopwatch build_watch;
  if (!c->add_batch(ids, base).ok()) {
    std::cerr << "build failed\n";
    return 1;
  }
  const double build_seconds = build_watch.elapsed_seconds();
  base = {};
  const d::Stopwatch save_watch;
  const vf::Status saved = c->save(o.index_file);
  const double save_seconds = save_watch.elapsed_seconds();
  if (!saved.ok()) {
    std::cerr << saved.to_string() << "\n";
    return 1;
  }
  const auto bytes = std::filesystem::file_size(o.index_file);
  std::ostringstream json;
  json << "{\n  \"schema\": 1, \"suite\": \"storage-save\",\n"
       << environment_json() << "  \"dataset\": {\"distribution\": \"" << o.distribution
       << "\", \"n\": " << o.n << ", \"dim\": " << o.dim << ", \"metric\": \""
       << vf::to_string(metric) << "\", \"seed\": " << o.seed << "},\n  \"params\": {\"index\": \""
       << o.index_type << "\", \"M\": " << o.m << ", \"ef_construction\": " << o.ef_construction
       << ", \"threads\": 1},\n"
       << "  \"result\": {\"build_seconds\": " << build_seconds
       << ", \"save_seconds\": " << save_seconds << ", \"file_bytes\": " << bytes
       << ", \"save_mib_per_s\": " << static_cast<double>(bytes) / (1024.0 * 1024.0) / save_seconds
       << "}\n}\n";
  emit(o, json.str());
  return 0;
}

// load: open an existing index in one mode, then query it twice (first pass includes page faults
// for mmap loads; second pass is steady state).
int run_load(const Options& o) {
  if (o.index_file.empty()) {
    std::cerr << "--index-file is required\n";
    return 2;
  }
  const bool mmap = o.load_mode == "mmap";
  const vf::bench::ProcessMemory before = vf::bench::process_memory();
  const d::Stopwatch open_watch;
  vf::Result<std::unique_ptr<vf::Collection>> loaded =
      vf::Collection::load(o.index_file, {.use_mmap = mmap, .prefault = o.prefault});
  const double open_seconds = open_watch.elapsed_seconds();
  if (!loaded.ok()) {
    std::cerr << loaded.status().to_string() << "\n";
    return 1;
  }
  const vf::bench::ProcessMemory after_open = vf::bench::process_memory();
  const vf::Collection& c = *loaded.value();
  Options qo = o;
  qo.dim = c.config().dim;
  const std::vector<float> queries = generate(qo, o.queries, o.seed + 1);
  const auto nq = static_cast<std::size_t>(o.queries);
  vf::SearchParams params;
  params.k = o.k;
  params.ef_search = o.ef_search.front();
  std::vector<Neighbor> out(o.k);
  std::array<std::vector<double>, 2> passes;
  std::array<double, 2> totals{};
  for (std::size_t pass = 0; pass < 2; ++pass) {
    passes[pass].resize(nq);
    for (std::size_t q = 0; q < nq; ++q) {
      const d::Stopwatch sw;
      static_cast<void>(
          c.search_into(std::span<const float>(queries).subspan(q * qo.dim, qo.dim), params, out));
      passes[pass][q] = sw.elapsed_seconds();
      totals[pass] += passes[pass][q];
    }
  }
  const double first_query_us = passes[0][0] * 1e6;
  const vf::bench::ProcessMemory after_queries = vf::bench::process_memory();
  std::ostringstream json;
  json << "{\n  \"schema\": 1, \"suite\": \"storage-load\",\n"
       << environment_json()
       << "  \"params\": {\"file_bytes\": " << std::filesystem::file_size(o.index_file)
       << ", \"index\": \"" << vf::to_string(c.config().index)
       << "\", \"rows\": " << c.stats().row_count << ", \"dim\": " << c.config().dim
       << ", \"load_mode\": \"" << o.load_mode
       << "\", \"prefault\": " << (o.prefault ? "true" : "false") << ", \"k\": " << o.k
       << ", \"ef_search\": " << o.ef_search.front() << ", \"queries\": " << nq
       << ", \"page_cache\": \"warm (not evicted)\"},\n  \"result\": {\"open_seconds\": "
       << open_seconds << ", \"first_query_us\": " << first_query_us;
  for (std::size_t pass = 0; pass < 2; ++pass) {
    std::sort(passes[pass].begin(), passes[pass].end());
    json << ", \"pass" << pass + 1 << "\": {\"qps\": " << static_cast<double>(nq) / totals[pass]
         << ", \"p50_us\": " << percentile(passes[pass], 50) * 1e6
         << ", \"p99_us\": " << percentile(passes[pass], 99) * 1e6
         << ", \"max_us\": " << passes[pass].back() * 1e6 << "}";
  }
  json << ", \"rss_bytes\": {\"before\": " << before.rss_bytes
       << ", \"after_open\": " << after_open.rss_bytes
       << ", \"after_queries\": " << after_queries.rss_bytes
       << "}, \"private_bytes\": {\"before\": " << before.private_bytes
       << ", \"after_open\": " << after_open.private_bytes
       << ", \"after_queries\": " << after_queries.private_bytes
       << "}, \"mapped_vector_bytes\": " << c.stats().memory.mapped_vectors_bytes << "}\n}\n";
  emit(o, json.str());
  return 0;
}

std::string json_escape(const std::string& s) {
  std::string out;
  for (const char c : s) {
    if (c == '"' || c == '\\') {
      out += '\\';
    }
    out += c;
  }
  return out;
}

// threads: batch-search throughput of a loaded index as a function of the thread count
// (docs/DESIGN.md §16.3 "threads"). Each thread count runs the whole query set `repeat` times.
int run_threads(const Options& o) {
  if (o.index_file.empty()) {
    std::cerr << "--index-file is required\n";
    return 2;
  }
  vf::Result<std::unique_ptr<vf::Collection>> loaded =
      vf::Collection::load(o.index_file, {.use_mmap = false});
  if (!loaded.ok()) {
    std::cerr << loaded.status().to_string() << "\n";
    return 1;
  }
  const vf::Collection& c = *loaded.value();
  Options qo = o;
  qo.dim = c.config().dim;
  std::vector<float> queries;
  std::size_t nq = static_cast<std::size_t>(o.queries);
  if (o.queries_file.empty()) {
    queries = generate(qo, o.queries, o.seed + 1);
  } else {
    vf::Result<d::Matrix<float>> file = d::read_float_matrix(o.queries_file);
    if (!file.ok() || file.value().cols != c.config().dim) {
      std::cerr << "cannot use --queries-file: "
                << (file.ok() ? std::string("dimension mismatch") : file.status().to_string())
                << '\n';
      return 1;
    }
    nq = static_cast<std::size_t>(file.value().rows);
    queries = std::move(file.value().data);
  }
  vf::SearchParams params;
  params.k = o.k;
  params.ef_search = o.ef_search.front();
  std::vector<ExternalId> ids(nq * o.k);
  std::vector<float> distances(nq * o.k);
  std::vector<std::uint32_t> counts(nq);
  std::ostringstream json;
  json << "{\n  \"schema\": 1, \"suite\": \"threads\",\n"
       << environment_json() << "  \"params\": {\"index\": \"" << vf::to_string(c.config().index)
       << "\", \"rows\": " << c.size() << ", \"dim\": " << c.config().dim << ", \"metric\": \""
       << vf::to_string(c.config().metric) << "\", \"k\": " << o.k
       << ", \"ef_search\": " << o.ef_search.front() << ", \"queries\": " << nq
       << ", \"repeat\": " << o.repeat
       << ", \"hardware_threads\": " << std::thread::hardware_concurrency() << "},\n"
       << "  \"results\": [\n";
  double single_qps = 0.0;
  for (std::size_t t = 0; t < o.threads.size(); ++t) {
    const std::uint32_t threads = o.threads[t];
    const std::unique_ptr<vf::ThreadPool> pool =
        threads > 1 ? std::make_unique<vf::ThreadPool>(threads - 1) : nullptr;
    // Warm-up: sizes the search contexts for this thread count.
    if (!c.search_batch(queries, nq, params, ids, distances, counts, pool.get()).ok()) {
      std::cerr << "search_batch failed\n";
      return 1;
    }
    std::vector<double> rounds;
    for (std::uint32_t r = 0; r < o.repeat; ++r) {
      const d::Stopwatch sw;
      static_cast<void>(c.search_batch(queries, nq, params, ids, distances, counts, pool.get()));
      rounds.push_back(static_cast<double>(nq) / sw.elapsed_seconds());
    }
    std::sort(rounds.begin(), rounds.end());
    const double qps = rounds[rounds.size() / 2];
    if (threads == 1) {
      single_qps = qps;
    }
    json << "    {\"threads\": " << threads << ", \"qps_median\": " << qps
         << ", \"qps_min\": " << rounds.front() << ", \"qps_max\": " << rounds.back();
    if (single_qps > 0.0) {
      json << ", \"speedup\": " << qps / single_qps;
    }
    json << "}" << (t + 1 < o.threads.size() ? "," : "") << "\n";
    std::cerr << "threads=" << threads << " qps=" << qps << "\n";
  }
  json << "  ]\n}\n";
  emit(o, json.str());
  return 0;
}

// ingest: search latency of one reader thread while a writer adds vectors with add_batch
// (docs/DESIGN.md §16.3 "ingest"), compared with the same reader on the idle collection.
int run_ingest(const Options& o, vf::Metric metric) {
  vf::CollectionConfig cfg;
  cfg.dim = o.dim;
  cfg.metric = metric;
  cfg.index = o.index_type == "flat" ? vf::IndexType::Flat : vf::IndexType::Hnsw;
  cfg.hnsw.M = o.m;
  cfg.hnsw.ef_construction = o.ef_construction;
  cfg.concurrency = concurrency_of(o);
  const std::unique_ptr<vf::Collection> c = vf::Collection::create(cfg).value();
  const std::vector<float> base = generate(o, o.n, o.seed);
  const std::vector<float> queries = generate(o, o.queries, o.seed + 1);
  const auto half = static_cast<std::size_t>(o.n / 2);
  std::vector<ExternalId> ids(static_cast<std::size_t>(o.n));
  for (std::size_t i = 0; i < ids.size(); ++i) {
    ids[i] = i;
  }
  std::cerr << "preloading " << half << " vectors\n";
  if (!c->add_batch(std::span<const ExternalId>(ids).first(half),
                    std::span<const float>(base).first(half * o.dim))
           .ok()) {
    std::cerr << "preload failed\n";
    return 1;
  }
  vf::SearchParams params;
  params.k = o.k;
  params.ef_search = o.ef_search.front();
  const auto nq = static_cast<std::size_t>(o.queries);

  auto measure = [&](const std::atomic<bool>& keep_going, std::size_t min_queries) {
    std::vector<double> latencies;
    std::vector<vf::Neighbor> out(o.k);
    for (std::size_t q = 0; keep_going.load() || q < min_queries; ++q) {
      const d::Stopwatch sw;
      static_cast<void>(c->search_into(
          std::span<const float>(queries).subspan((q % nq) * o.dim, o.dim), params, out));
      latencies.push_back(sw.elapsed_seconds());
    }
    std::sort(latencies.begin(), latencies.end());
    return latencies;
  };

  const std::atomic<bool> idle{false};
  const std::vector<double> quiet = measure(idle, nq);

  std::atomic<bool> writing{true};
  double ingest_seconds = 0.0;
  std::thread writer([&] {
    const std::unique_ptr<vf::ThreadPool> pool =
        o.writer_threads > 1 ? std::make_unique<vf::ThreadPool>(o.writer_threads - 1) : nullptr;
    const d::Stopwatch sw;
    for (std::size_t first = half; first < ids.size(); first += o.batch) {
      const std::size_t count = std::min<std::size_t>(o.batch, ids.size() - first);
      static_cast<void>(c->add_batch(
          std::span<const ExternalId>(ids).subspan(first, count),
          std::span<const float>(base).subspan(first * o.dim, count * o.dim), {}, pool.get()));
    }
    ingest_seconds = sw.elapsed_seconds();
    writing = false;
  });
  const std::vector<double> busy = measure(writing, 1);
  writer.join();

  auto summary = [](const std::vector<double>& s) {
    std::ostringstream out;
    out << "{\"queries\": " << s.size() << ", \"p50_us\": " << percentile(s, 50) * 1e6
        << ", \"p99_us\": " << percentile(s, 99) * 1e6
        << ", \"p999_us\": " << percentile(s, 99.9) * 1e6 << ", \"max_us\": " << s.back() * 1e6
        << "}";
    return out.str();
  };
  std::ostringstream json;
  json << "{\n  \"schema\": 1, \"suite\": \"ingest\",\n"
       << environment_json() << "  \"dataset\": {\"distribution\": \"" << o.distribution
       << "\", \"n\": " << o.n << ", \"dim\": " << o.dim << ", \"metric\": \""
       << vf::to_string(metric) << "\", \"seed\": " << o.seed << "},\n"
       << "  \"params\": {\"index\": \"" << o.index_type << "\", \"M\": " << o.m
       << ", \"ef_construction\": " << o.ef_construction << ", \"k\": " << o.k
       << ", \"ef_search\": " << o.ef_search.front() << ", \"preloaded\": " << half
       << ", \"ingested\": " << (ids.size() - half) << ", \"batch\": " << o.batch
       << ", \"concurrency\": \"" << o.concurrency << "\", \"writer_threads\": " << o.writer_threads
       << ", \"reader_threads\": 1},\n"
       << "  \"result\": {\"ingest_seconds\": " << ingest_seconds
       << ", \"ingest_vectors_per_s\": " << static_cast<double>(ids.size() - half) / ingest_seconds
       << ", \"search_idle\": " << summary(quiet) << ", \"search_during_ingest\": " << summary(busy)
       << "}\n}\n";
  emit(o, json.str());
  return 0;
}

// build: HNSW construction time through Collection::add_batch as a function of the thread count
// (docs/DESIGN.md §16.3 "build scaling"), with recall@k of each graph against exact results and
// the validator's view of each graph. The first row is the Level A (Coarse) serial build.
int run_build(const Options& o, vf::Metric metric) {
  const std::vector<float> base = generate(o, o.n, o.seed);
  const std::vector<float> queries = generate(o, o.queries, o.seed + 1);
  const auto n = static_cast<std::size_t>(o.n);
  const auto nq = static_cast<std::size_t>(o.queries);
  std::vector<ExternalId> ids(n);
  for (std::size_t i = 0; i < n; ++i) {
    ids[i] = i;
  }
  vf::CollectionConfig cfg;
  cfg.dim = o.dim;
  cfg.metric = metric;
  cfg.hnsw.M = o.m;
  cfg.hnsw.ef_construction = o.ef_construction;

  // Exact ground truth, computed in parallel.
  const auto hw = std::max<std::uint32_t>(std::thread::hardware_concurrency(), 2U);
  vf::ThreadPool gt_pool(hw - 1);
  vf::CollectionConfig flat_cfg = cfg;
  flat_cfg.index = vf::IndexType::Flat;
  const std::unique_ptr<vf::Collection> flat = vf::Collection::create(flat_cfg).value();
  static_cast<void>(flat->add_batch(ids, base));
  vf::SearchParams exact;
  exact.k = o.k;
  std::vector<ExternalId> gt_ids(nq * o.k);
  std::vector<float> gt_dist(nq * o.k);
  std::vector<std::uint32_t> counts(nq);
  static_cast<void>(flat->search_batch(queries, nq, exact, gt_ids, gt_dist, counts, &gt_pool));

  std::ostringstream json;
  json << "{\n  \"schema\": 1, \"suite\": \"build\",\n"
       << environment_json() << "  \"dataset\": {\"distribution\": \"" << o.distribution
       << "\", \"n\": " << o.n << ", \"dim\": " << o.dim << ", \"metric\": \""
       << vf::to_string(metric) << "\", \"seed\": " << o.seed << "},\n"
       << "  \"params\": {\"M\": " << o.m << ", \"ef_construction\": " << o.ef_construction
       << ", \"k\": " << o.k << ", \"queries\": " << nq << ", \"batch\": " << o.batch
       << ", \"hardware_threads\": " << std::thread::hardware_concurrency() << "},\n"
       << "  \"results\": [\n";

  struct Run {
    vf::Concurrency mode;
    std::uint32_t threads;
  };
  std::vector<Run> runs{{vf::Concurrency::Coarse, 1}};
  for (const std::uint32_t t : o.threads) {
    runs.push_back({vf::Concurrency::Concurrent, t});
  }
  double coarse_seconds = 0.0;
  for (std::size_t r = 0; r < runs.size(); ++r) {
    cfg.concurrency = runs[r].mode;
    const std::unique_ptr<vf::Collection> c = vf::Collection::create(cfg).value();
    const std::unique_ptr<vf::ThreadPool> pool =
        runs[r].threads > 1 ? std::make_unique<vf::ThreadPool>(runs[r].threads - 1) : nullptr;
    const d::Stopwatch sw;
    for (std::size_t first = 0; first < n; first += o.batch) {
      const std::size_t count = std::min<std::size_t>(o.batch, n - first);
      if (!c->add_batch(std::span<const ExternalId>(ids).subspan(first, count),
                        std::span<const float>(base).subspan(first * o.dim, count * o.dim), {},
                        pool.get())
               .ok()) {
        std::cerr << "add_batch failed\n";
        return 1;
      }
    }
    const double seconds = sw.elapsed_seconds();
    if (r == 0) {
      coarse_seconds = seconds;
    }
    const auto& hnsw = static_cast<const d::HnswBackend&>(*d::CollectionFactory::state(*c).backend);
    const d::HnswValidator validator(hnsw.graph());
    const bool valid = validator.check_invariants().ok();
    const std::uint64_t unreachable = valid ? validator.reachability().unreachable_level0() : 0;
    json << "    {\"concurrency\": \"" << vf::to_string(runs[r].mode)
         << "\", \"threads\": " << runs[r].threads << ", \"build_seconds\": " << seconds
         << ", \"speedup_vs_coarse\": " << coarse_seconds / seconds
         << ", \"valid\": " << (valid ? "true" : "false")
         << ", \"unreachable_level0\": " << unreachable
         << ", \"orphans_unrepaired\": " << hnsw.build_stats().orphans_unrepaired
         << ", \"recall\": [";
    for (std::size_t e = 0; e < o.ef_search.size(); ++e) {
      vf::SearchParams p;
      p.k = o.k;
      p.ef_search = o.ef_search[e];
      std::vector<ExternalId> got_ids(nq * o.k);
      std::vector<float> got_dist(nq * o.k);
      static_cast<void>(c->search_batch(queries, nq, p, got_ids, got_dist, counts, &gt_pool));
      const double recall =
          d::recall_at_k(got_ids, got_dist, o.k, gt_ids, gt_dist, o.k, nq, o.k).value().mean;
      json << (e == 0 ? "" : ", ") << "{\"ef_search\": " << o.ef_search[e]
           << ", \"recall\": " << recall << "}";
    }
    json << "]}" << (r + 1 < runs.size() ? "," : "") << "\n";
    std::cerr << vf::to_string(runs[r].mode) << " threads=" << runs[r].threads
              << " build=" << seconds << "s valid=" << valid << "\n";
  }
  json << "  ]\n}\n";
  emit(o, json.str());
  return 0;
}

int run(int argc, char** argv) {
  Options o;
  CLI::App app{"vf_bench: HNSW build + ef_search sweep against exact ground truth"};
  app.add_option("--n", o.n, "base vectors")->check(CLI::Range(1ULL, 100000000ULL));
  app.add_option("--queries", o.queries, "query vectors")->check(CLI::Range(1ULL, 10000000ULL));
  app.add_option("--dim", o.dim, "dimension")->check(CLI::Range(1U, vf::kMaxDim));
  app.add_option("--metric", o.metric, "l2 | ip | cosine");
  app.add_option("--distribution", o.distribution, "gmm | uniform")
      ->check(CLI::IsMember({"gmm", "uniform"}));
  app.add_option("--clusters", o.clusters, "mixture components");
  app.add_option("--spread", o.spread, "mixture standard deviation");
  app.add_option("--seed", o.seed, "dataset seed (base uses seed, queries seed+1)");
  app.add_option("--M", o.m, "HNSW M");
  app.add_option("--ef-construction", o.ef_construction, "HNSW ef_construction");
  app.add_option("--ef-search", o.ef_search, "ef_search values")->delimiter(',');
  app.add_option("--k", o.k, "neighbours per query")->check(CLI::Range(1U, 1000U));
  app.add_option("--selection", o.selection, "heuristic | simple")
      ->check(CLI::IsMember({"heuristic", "simple"}));
  app.add_flag("--no-repair", o.no_repair, "disable new-node orphan repair");
  app.add_flag("--skip-flat-timing", o.skip_flat_timing, "do not time Flat queries");
  app.add_option("--out", o.out, "JSON output file (default: stdout only)");
  app.add_option("--scenario", o.scenario, "sweep | save | load | threads | ingest | build")
      ->check(CLI::IsMember({"sweep", "save", "load", "threads", "ingest", "build"}));
  app.add_option("--concurrency", o.concurrency, "ingest/build: coarse | concurrent")
      ->check(CLI::IsMember({"coarse", "concurrent"}));
  app.add_option("--queries-file", o.queries_file, "threads: query vectors (.npy or .fvecs)");
  app.add_option("--writer-threads", o.writer_threads, "ingest: threads of the add_batch writer")
      ->check(CLI::Range(1U, 256U));
  app.add_option("--threads", o.threads, "threads: thread counts")->delimiter(',');
  app.add_option("--repeat", o.repeat, "threads: timed passes per thread count")
      ->check(CLI::Range(1U, 1000U));
  app.add_option("--batch", o.batch, "ingest: vectors per add_batch call")
      ->check(CLI::Range(1U, 10000000U));
  app.add_option("--index-file", o.index_file, "save/load: index file path");
  app.add_option("--index", o.index_type, "save/ingest: flat | hnsw")
      ->check(CLI::IsMember({"flat", "hnsw"}));
  app.add_option("--load-mode", o.load_mode, "load: heap | mmap")
      ->check(CLI::IsMember({"heap", "mmap"}));
  app.add_flag("--prefault", o.prefault, "load: prefault mapped vectors");
  app.add_option("--simd", o.simd, "kernel tier request, sets VF_SIMD: auto | scalar | avx2")
      ->check(CLI::IsMember({"auto", "scalar", "avx2"}));
  app.add_option("--prefetch", o.prefetch, "sweep: HNSW neighbour-row prefetch: on | off")
      ->check(CLI::IsMember({"on", "off"}));
  CLI11_PARSE(app, argc, argv);

  // The tier is resolved on first kernel use, which happens after this point.
  if (!o.simd.empty()) {
    set_simd_request(o.simd);
  }
  if (const vf::Status simd = vf::simd_status(); !simd.ok()) {
    std::cerr << simd.to_string() << "\n";
    return 2;
  }
  std::cerr << "simd tier " << vf::to_string(vf::active_simd_level()) << " (" << d::kernels().name
            << ")\n";

  vf::Result<vf::Metric> metric = vf::parse_metric(o.metric);
  if (!metric.ok()) {
    std::cerr << metric.status().to_string() << "\n";
    return 2;
  }
  if (o.scenario == "save") {
    return run_save(o, metric.value());
  }
  if (o.scenario == "load") {
    return run_load(o);
  }
  if (o.scenario == "threads") {
    return run_threads(o);
  }
  if (o.scenario == "ingest") {
    return run_ingest(o, metric.value());
  }
  if (o.scenario == "build") {
    return run_build(o, metric.value());
  }
  vf::HnswParams params;
  params.M = o.m;
  params.ef_construction = o.ef_construction;
  if (const vf::Status st = params.validate(); !st.ok()) {
    std::cerr << st.to_string() << "\n";
    return 2;
  }
  const d::KernelTable& kernels = d::kernels();

  // Index: loaded from --index-file (the same graph for every tier) or built here.
  const bool from_file = !o.index_file.empty();
  std::unique_ptr<vf::Collection> loaded;
  d::VectorStore store = d::VectorStore::create({.dim = o.dim}).value();
  d::TombstoneSet deleted;
  std::unique_ptr<d::HnswBackend> built;
  d::HnswBackend* hnsw = nullptr;
  bool normalized = false;
  std::vector<float> base;
  double build_seconds = 0.0;
  if (from_file) {
    vf::Result<std::unique_ptr<vf::Collection>> opened =
        vf::Collection::load(o.index_file, {.use_mmap = false});
    if (!opened.ok()) {
      std::cerr << opened.status().to_string() << "\n";
      return 1;
    }
    loaded = std::move(opened).value();
    const d::CollectionState& state = d::CollectionFactory::state(*loaded);
    if (loaded->config().index != vf::IndexType::Hnsw || state.deleted.any()) {
      std::cerr << "--index-file must be an HNSW index without deletions\n";
      return 2;
    }
    hnsw = &static_cast<d::HnswBackend&>(*state.backend);
    o.dim = loaded->config().dim;
    o.n = state.vectors.size();
    o.m = hnsw->params().M;
    o.ef_construction = hnsw->params().ef_construction;
    metric = loaded->config().metric;
    normalized = state.normalized;
    base.resize(static_cast<std::size_t>(o.n) * o.dim);
    for (std::uint64_t r = 0; r < o.n; ++r) {
      const std::span<const float> row = state.vectors.row(static_cast<vf::InternalId>(r));
      std::copy(row.begin(), row.end(), base.begin() + static_cast<std::ptrdiff_t>(r * o.dim));
    }
    std::cerr << "loaded " << o.index_file << " (" << o.n << " x " << o.dim << ")\n";
  } else {
    normalized = metric.value() == vf::Metric::Cosine;
    std::cerr << "generating " << o.n << " vectors (dim " << o.dim << ")\n";
    base = generate(o, o.n, o.seed);
  }
  const std::vector<float> queries = generate(o, o.queries, o.seed + 1);
  const auto nq = static_cast<std::size_t>(o.queries);

  // Exact ground truth through a Flat collection (ids 0..n-1 == internal ids).
  std::cerr << "ground truth\n";
  vf::CollectionConfig flat_cfg;
  flat_cfg.dim = o.dim;
  flat_cfg.metric = metric.value();
  flat_cfg.index = vf::IndexType::Flat;
  const std::unique_ptr<vf::Collection> flat = vf::Collection::create(flat_cfg).value();
  std::vector<ExternalId> ids(static_cast<std::size_t>(o.n));
  for (std::size_t i = 0; i < ids.size(); ++i) {
    ids[i] = i;
  }
  if (!flat->add_batch(ids, base).ok()) {
    std::cerr << "flat build failed\n";
    return 1;
  }
  vf::SearchParams sp;
  sp.k = o.k;
  std::vector<ExternalId> gt_ids(nq * o.k);
  std::vector<float> gt_dist(nq * o.k);
  std::vector<std::uint32_t> gt_counts(nq);
  if (!flat->search_batch(queries, nq, sp, gt_ids, gt_dist, gt_counts).ok()) {
    std::cerr << "ground truth failed\n";
    return 1;
  }
  double flat_seconds = 0.0;
  if (!o.skip_flat_timing) {
    std::vector<Neighbor> out(o.k);
    const d::Stopwatch sw;
    for (std::size_t q = 0; q < nq; ++q) {
      static_cast<void>(
          flat->search_into(std::span<const float>(queries).subspan(q * o.dim, o.dim), sp, out));
    }
    flat_seconds = sw.elapsed_seconds();
  }

  if (!from_file) {
    std::cerr << "building HNSW (M=" << o.m << ", ef_construction=" << o.ef_construction << ")\n";
    store = d::VectorStore::create({.dim = o.dim}).value();
    d::HnswBuildOptions build_options;
    build_options.selection =
        o.selection == "simple" ? d::NeighborSelection::Simple : d::NeighborSelection::Heuristic;
    build_options.repair_orphans = !o.no_repair;
    built = d::HnswBackend::create(store, deleted, metric.value(), normalized, kernels, params,
                                   build_options)
                .value();
    hnsw = built.get();
    hnsw->set_search_options({.prefetch = o.prefetch == "on"});
    if (normalized) {
      for (std::size_t r = 0; r < static_cast<std::size_t>(o.n); ++r) {
        static_cast<void>(
            d::normalize_inplace(std::span<float>(base).subspan(r * o.dim, o.dim), kernels));
      }
    }
    if (!store.reserve(o.n).ok()) {
      return 1;
    }
    deleted.ensure_size(o.n);
    const d::Stopwatch build_watch;
    for (std::size_t r = 0; r < static_cast<std::size_t>(o.n); ++r) {
      const vf::InternalId id =
          store.append(std::span<const float>(base).subspan(r * o.dim, o.dim)).value();
      if (!hnsw->add(id).ok()) {
        std::cerr << "hnsw add failed\n";
        return 1;
      }
    }
    build_seconds = build_watch.elapsed_seconds();
  }
  if (from_file) {
    hnsw->set_search_options({.prefetch = o.prefetch == "on"});
  }
  const d::HnswValidator validator(hnsw->graph());
  const vf::Status invariants = validator.check_invariants();
  const d::HnswReachability reach = validator.reachability();
  std::cerr << "built in " << build_seconds << " s; invariants " << invariants.to_string()
            << "; unreachable level0 " << reach.unreachable_level0() << "\n";

  std::ostringstream json;
  const d::CpuFeatures& cpu = d::cpu_features();
  json << "{\n  \"schema\": 1, \"suite\": \"hnsw-sweep-minimal\",\n"
       << "  \"git\": {\"sha\": \"" << vf::kGitSha << "\"},\n"
       << "  \"machine\": {\"cpu\": \"" << json_escape(cpu.brand) << "\"},\n"
       << "  \"build\": {\"compiler\": \"" << vf::kCompiler << "\", \"type\": \"" << vf::kBuildType
       << "\", \"simd_tier\": \"" << vf::to_string(vf::active_simd_level())
       << "\", \"kernel_table\": \"" << kernels.name << "\"},\n"
       << "  \"dataset\": {\"distribution\": \"" << o.distribution << "\", \"n\": " << o.n
       << ", \"queries\": " << o.queries << ", \"dim\": " << o.dim << ", \"metric\": \""
       << vf::to_string(metric.value()) << "\", \"clusters\": " << o.clusters
       << ", \"spread\": " << o.spread << ", \"seed\": " << o.seed << "},\n"
       << "  \"params\": {\"index\": \"hnsw\", \"M\": " << o.m
       << ", \"ef_construction\": " << o.ef_construction << ", \"k\": " << o.k
       << ", \"selection\": \"" << o.selection
       << "\", \"repair_orphans\": " << (o.no_repair ? "false" : "true")
       << ", \"prefetch\": " << (o.prefetch == "on" ? "true" : "false")
       << ", \"index_file\": " << (from_file ? "true" : "false") << ", \"threads\": 1},\n"
       << "  \"build_result\": {\"seconds\": " << build_seconds
       << ", \"distance_computations\": " << hnsw->build_stats().distance_computations
       << ", \"orphan_repairs\": " << hnsw->build_stats().orphan_repairs
       << ", \"orphans_unrepaired\": " << hnsw->build_stats().orphans_unrepaired
       << ", \"index_bytes\": " << hnsw->stats().index_bytes
       << ", \"invariants_ok\": " << (invariants.ok() ? "true" : "false")
       << ", \"unreachable_level0\": " << reach.unreachable_level0() << "},\n";
  if (!o.skip_flat_timing) {
    json << "  \"flat\": {\"seconds\": " << flat_seconds
         << ", \"qps\": " << static_cast<double>(nq) / flat_seconds
         << ", \"mean_latency_us\": " << flat_seconds * 1e6 / static_cast<double>(nq) << "},\n";
  }
  json << "  \"sweep\": [\n";

  d::SearchContext context;
  std::vector<Neighbor> out(o.k);
  std::vector<ExternalId> approx_ids(nq * o.k);
  std::vector<float> approx_dist(nq * o.k);
  std::vector<double> latencies(nq);
  for (std::size_t e = 0; e < o.ef_search.size(); ++e) {
    const d::SearchKnobs knobs{.k = o.k, .ef = o.ef_search[e]};
    auto query_view = [&](std::size_t q) {
      const auto row = std::span<const float>(queries).subspan(q * o.dim, o.dim);
      d::QueryView view{.data = row.data(), .inv_norm = 1.0F};
      if (normalized) {
        view.inv_norm = d::inverse_norm(row, kernels).value();
      }
      return view;
    };
    for (std::size_t q = 0; q < nq; ++q) {  // warm-up pass, not recorded
      static_cast<void>(hnsw->search_with(context, query_view(q), knobs, out));
    }
    context.distance_computations = 0;
    double total_seconds = 0.0;
    for (std::size_t q = 0; q < nq; ++q) {
      const d::QueryView view = query_view(q);
      const d::Stopwatch sw;
      const std::size_t count = hnsw->search_with(context, view, knobs, out);
      latencies[q] = sw.elapsed_seconds();
      total_seconds += latencies[q];
      for (std::size_t j = 0; j < o.k; ++j) {
        approx_ids[(q * o.k) + j] = j < count ? out[j].id : vf::kInvalidExternalId;
        approx_dist[(q * o.k) + j] = j < count ? out[j].distance : INFINITY;
      }
    }
    const d::RecallStats recall =
        d::recall_at_k(approx_ids, approx_dist, o.k, gt_ids, gt_dist, o.k, nq, o.k).value();
    std::sort(latencies.begin(), latencies.end());
    json << "    {\"ef_search\": " << o.ef_search[e] << ", \"recall_at_k\": " << recall.mean
         << ", \"recall_min\": " << recall.min
         << ", \"qps\": " << static_cast<double>(nq) / total_seconds
         << ", \"latency_us\": {\"p50\": " << percentile(latencies, 50) * 1e6
         << ", \"p95\": " << percentile(latencies, 95) * 1e6
         << ", \"p99\": " << percentile(latencies, 99) * 1e6
         << ", \"mean\": " << total_seconds * 1e6 / static_cast<double>(nq)
         << ", \"max\": " << latencies.back() * 1e6 << "}, \"dist_comps_mean\": "
         << static_cast<double>(context.distance_computations) / static_cast<double>(nq) << "}"
         << (e + 1 < o.ef_search.size() ? "," : "") << "\n";
    std::cerr << "ef=" << o.ef_search[e] << " recall=" << recall.mean
              << " qps=" << static_cast<double>(nq) / total_seconds << "\n";
  }
  json << "  ]\n}\n";

  std::cout << json.str();
  if (!o.out.empty()) {
    std::ofstream file(o.out, std::ios::binary);
    file << json.str();
    if (!file) {
      std::cerr << "cannot write " << o.out << "\n";
      return 1;
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "vf_bench: " << e.what() << "\n";
  } catch (...) {
    std::cerr << "vf_bench: unknown exception\n";
  }
  return 2;
}
