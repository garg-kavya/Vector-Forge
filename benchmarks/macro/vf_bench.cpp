// vf_bench (minimal, Phase 3): build an HNSW index over a seeded synthetic dataset, sweep ef_search
// and report recall@k, latency, QPS and distance computations per query against exact (Flat)
// ground truth, plus Flat latency on the same queries. One configuration per process
// (docs/DESIGN.md §16). Single thread, active kernel tier.
//
// The index is built through HnswBackend directly (the same code path Collection uses, minus id
// mapping) so that graph statistics and distance counters are observable.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <vectorforge/collection.hpp>
#include <vectorforge/simd.hpp>
#include <vectorforge/version.hpp>

#include "core/vector_ops.hpp"
#include "index/hnsw/hnsw_backend.hpp"
#include "index/hnsw/hnsw_validator.hpp"
#include "process_memory.hpp"
#include "simd/cpu_features.hpp"
#include "simd/kernels.hpp"
#include "storage/tombstones.hpp"
#include "storage/vector_store.hpp"
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
};

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
       << "\", \"simd_tier\": \"" << vf::to_string(vf::active_simd_level()) << "\"},\n";
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
  app.add_option("--scenario", o.scenario, "sweep | save | load")
      ->check(CLI::IsMember({"sweep", "save", "load"}));
  app.add_option("--index-file", o.index_file, "save/load: index file path");
  app.add_option("--index", o.index_type, "save: flat | hnsw")
      ->check(CLI::IsMember({"flat", "hnsw"}));
  app.add_option("--load-mode", o.load_mode, "load: heap | mmap")
      ->check(CLI::IsMember({"heap", "mmap"}));
  app.add_flag("--prefault", o.prefault, "load: prefault mapped vectors");
  CLI11_PARSE(app, argc, argv);

  const vf::Result<vf::Metric> metric = vf::parse_metric(o.metric);
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
  vf::HnswParams params;
  params.M = o.m;
  params.ef_construction = o.ef_construction;
  if (const vf::Status st = params.validate(); !st.ok()) {
    std::cerr << st.to_string() << "\n";
    return 2;
  }
  const bool normalized = metric.value() == vf::Metric::Cosine;
  const d::KernelTable& kernels = d::kernels();

  std::cerr << "generating " << o.n << " + " << o.queries << " vectors (dim " << o.dim << ")\n";
  std::vector<float> base = generate(o, o.n, o.seed);
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

  // HNSW build.
  std::cerr << "building HNSW (M=" << o.m << ", ef_construction=" << o.ef_construction << ")\n";
  d::VectorStore store = d::VectorStore::create({.dim = o.dim}).value();
  d::TombstoneSet deleted;
  d::HnswBuildOptions build_options;
  build_options.selection =
      o.selection == "simple" ? d::NeighborSelection::Simple : d::NeighborSelection::Heuristic;
  build_options.repair_orphans = !o.no_repair;
  const std::unique_ptr<d::HnswBackend> hnsw =
      d::HnswBackend::create(store, deleted, metric.value(), normalized, kernels, params,
                             build_options)
          .value();
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
  const double build_seconds = build_watch.elapsed_seconds();
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
       << "\", \"simd_tier\": \"" << vf::to_string(vf::active_simd_level()) << "\"},\n"
       << "  \"dataset\": {\"distribution\": \"" << o.distribution << "\", \"n\": " << o.n
       << ", \"queries\": " << o.queries << ", \"dim\": " << o.dim << ", \"metric\": \""
       << vf::to_string(metric.value()) << "\", \"clusters\": " << o.clusters
       << ", \"spread\": " << o.spread << ", \"seed\": " << o.seed << "},\n"
       << "  \"params\": {\"index\": \"hnsw\", \"M\": " << o.m
       << ", \"ef_construction\": " << o.ef_construction << ", \"k\": " << o.k
       << ", \"selection\": \"" << o.selection
       << "\", \"repair_orphans\": " << (o.no_repair ? "false" : "true") << ", \"threads\": 1},\n"
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
