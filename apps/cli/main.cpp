// vectorforge command-line interface.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <map>
#include <new>
#include <string>

#include <vectorforge/simd.hpp>
#include <vectorforge/version.hpp>

#include "commands.hpp"

#include <CLI/CLI.hpp>

namespace {

int report(const vf::Status& status) {
  if (status.ok()) {
    return 0;
  }
  std::cerr << "error: " << status.to_string() << '\n';
  return 1;
}

int run(int argc, char** argv);

#if defined(VF_HAVE_SERVER)
std::string environment(const char* name) {
#if defined(_MSC_VER)
  char* value = nullptr;
  std::size_t length = 0;
  if (_dupenv_s(&value, &length, name) != 0 || value == nullptr) {
    return {};
  }
  std::string out(value);
  std::free(value);  // NOLINT(cppcoreguidelines-no-malloc): _dupenv_s allocates with malloc.
  return out;
#else
  const char* value =
      std::getenv(name);  // NOLINT(concurrency-mt-unsafe): read before threads start
  return value == nullptr ? std::string{} : std::string(value);
#endif
}
#endif

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::bad_alloc&) {
    std::cerr << "error: out of memory\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
  } catch (...) {
    std::cerr << "error: unknown exception\n";
  }
  return 2;
}

namespace {

int run(int argc, char** argv) {
  CLI::App app{"VectorForge vector search engine tools"};
  app.set_version_flag("--version",
                       std::string(vf::kVersion) + " (" + std::string(vf::kGitSha) + ")");
  app.require_subcommand(1);

  // gen-data ---------------------------------------------------------------------------------
  vf::cli::GenDataOptions gen;
  std::string gen_distribution = "gaussian-mixture";
  std::string gen_format = "npy";
  CLI::App* gen_cmd = app.add_subcommand("gen-data", "Generate a seeded synthetic float32 dataset");
  gen_cmd->add_option("--n", gen.rows, "Number of vectors")->required()->check(CLI::PositiveNumber);
  gen_cmd->add_option("--dim", gen.spec.dim, "Dimensionality")
      ->required()
      ->check(CLI::Range(1, 65536));
  gen_cmd->add_option("--dist", gen_distribution, "Distribution")
      ->check(CLI::IsMember({"uniform", "gaussian-mixture"}))
      ->capture_default_str();
  gen_cmd->add_option("--clusters", gen.spec.clusters, "Mixture components")->capture_default_str();
  gen_cmd->add_option("--spread", gen.spec.spread, "Per-component standard deviation")
      ->capture_default_str();
  gen_cmd->add_option("--center-range", gen.spec.center_range, "Centres uniform in [-r, r)")
      ->capture_default_str();
  gen_cmd->add_option("--seed", gen.spec.seed, "Sample seed")->capture_default_str();
  gen_cmd
      ->add_option("--mixture-seed", gen.spec.mixture_seed,
                   "Seed for mixture centres (share between base and query sets)")
      ->capture_default_str();
  gen_cmd->add_option("--format", gen_format, "Output format")
      ->check(CLI::IsMember({"npy", "fvecs"}))
      ->capture_default_str();
  gen_cmd->add_option("--out", gen.out, "Output file")->required();

  // ground-truth -----------------------------------------------------------------------------
  vf::cli::GroundTruthOptions gt;
  std::string gt_metric = "l2";
  CLI::App* gt_cmd =
      app.add_subcommand("ground-truth", "Compute exact k-NN ground truth with a Flat index");
  gt_cmd->add_option("--base", gt.base, "Base vectors (.npy or .fvecs)")
      ->required()
      ->check(CLI::ExistingFile);
  gt_cmd->add_option("--queries", gt.queries, "Query vectors (.npy or .fvecs)")
      ->required()
      ->check(CLI::ExistingFile);
  gt_cmd->add_option("--metric", gt_metric, "Metric")
      ->check(CLI::IsMember({"l2", "ip", "inner_product", "cosine"}))
      ->capture_default_str();
  gt_cmd->add_option("--k", gt.k, "Neighbours per query")
      ->check(CLI::Range(1, 1 << 20))
      ->capture_default_str();
  gt_cmd
      ->add_option("--out", gt.out_prefix,
                   "Output prefix: writes <prefix>.ids.npy and <prefix>.distances.npy")
      ->required();
  gt_cmd->add_option("--threads", gt.threads, "Threads (0 = all hardware threads)")
      ->capture_default_str();

  // build ------------------------------------------------------------------------------------
  vf::cli::BuildOptions build;
  std::string build_metric = "l2";
  std::string build_index = "hnsw";
  CLI::App* build_cmd =
      app.add_subcommand("build", "Build an index from a dataset file and save it");
  build_cmd->add_option("--input", build.input, "Vectors (.npy or .fvecs, float32)")
      ->required()
      ->check(CLI::ExistingFile);
  build_cmd->add_option("--ids", build.ids, "External ids (.npy int64/uint64; default row numbers)")
      ->check(CLI::ExistingFile);
  build_cmd->add_option("--metric", build_metric, "Metric")
      ->check(CLI::IsMember({"l2", "ip", "inner_product", "cosine"}))
      ->capture_default_str();
  build_cmd->add_option("--index", build_index, "Index type")
      ->check(CLI::IsMember({"flat", "hnsw"}))
      ->capture_default_str();
  build_cmd->add_flag("--normalize", build.config.normalize, "Normalise vectors to unit length");
  build_cmd->add_option("--M", build.config.hnsw.M, "HNSW links per node")->capture_default_str();
  build_cmd->add_option("--ef-construction", build.config.hnsw.ef_construction, "HNSW build beam")
      ->capture_default_str();
  build_cmd->add_option("--ef-search", build.config.hnsw.ef_search, "Default HNSW query beam")
      ->capture_default_str();
  build_cmd->add_option("--seed", build.config.hnsw.seed, "HNSW level seed")->capture_default_str();
  build_cmd->add_option("--out", build.out, "Output index file (.vfidx)")->required();
  build_cmd->add_option("--threads", build.threads, "Threads (0 = all hardware threads)")
      ->capture_default_str();

  // search -----------------------------------------------------------------------------------
  vf::cli::SearchOptions search;
  std::uint32_t search_ef = 0;
  bool search_no_mmap = false;
  CLI::App* search_cmd = app.add_subcommand("search", "Query an index file");
  search_cmd->add_option("--index", search.index, "Index file (.vfidx)")
      ->required()
      ->check(CLI::ExistingFile);
  search_cmd->add_option("--queries", search.queries, "Queries (.npy or .fvecs, float32)")
      ->required()
      ->check(CLI::ExistingFile);
  search_cmd->add_option("--k", search.k, "Neighbours per query")
      ->check(CLI::Range(1, 1 << 20))
      ->capture_default_str();
  search_cmd->add_option("--ef", search_ef, "HNSW beam width (default: the index's ef_search)")
      ->check(CLI::Range(1, 1 << 20));
  search_cmd->add_flag("--no-mmap", search_no_mmap, "Copy vectors to memory instead of mapping");
  search_cmd->add_option("--out", search.out_prefix,
                         "Output prefix: writes <prefix>.ids.npy and <prefix>.distances.npy");
  search_cmd->add_option("--gt", search.ground_truth,
                         "Ground truth: prefix from ground-truth, or a .ivecs id file");

  // info / verify ----------------------------------------------------------------------------
  std::filesystem::path info_path;
  CLI::App* info_cmd = app.add_subcommand("info", "Describe an index file");
  info_cmd->add_option("index", info_path, "Index file (.vfidx)")
      ->required()
      ->check(CLI::ExistingFile);
  std::filesystem::path verify_path;
  CLI::App* verify_cmd =
      app.add_subcommand("verify", "Verify checksums, structure and graph invariants");
  verify_cmd->add_option("index", verify_path, "Index file (.vfidx)")
      ->required()
      ->check(CLI::ExistingFile);

#if defined(VF_HAVE_SERVER)
  // serve -------------------------------------------------------------------------------------
  vf::cli::ServeOptions serve;
  bool serve_list_routes = false;
  bool serve_no_mmap = false;
  std::size_t serve_max_body_mb = serve.server.limits.max_body_bytes >> 20U;
  std::uint32_t serve_drain_seconds = 10;
  CLI::App* serve_cmd = app.add_subcommand("serve", "Serve a catalog directory over HTTP");
  serve_cmd->add_option("--data-dir", serve.data_dir, "Catalog directory (created if missing)");
  serve_cmd->add_option("--host", serve.server.host, "Listen address")->capture_default_str();
  serve_cmd->add_option("--port", serve.server.port, "Listen port (0 = any free port)")
      ->capture_default_str()
      ->check(CLI::Range(0, 65535));
  serve_cmd->add_option("--http-threads", serve.server.http_threads,
                        "Connection threads (0 = hardware threads)");
  serve_cmd->add_option("--threads", serve.server.compute_threads,
                        "Compute threads for batch requests (0 = hardware threads)");
  serve_cmd->add_option("--api-key", serve.server.api_key,
                        "Require this bearer token on /v1 routes (default: $VF_API_KEY)");
  serve_cmd->add_option("--max-body-mb", serve_max_body_mb, "Request body limit (MiB)")
      ->capture_default_str()
      ->check(CLI::Range(1, 4096));
  serve_cmd->add_option("--max-batch", serve.server.limits.max_batch, "Vectors per request")
      ->capture_default_str()
      ->check(CLI::PositiveNumber);
  serve_cmd->add_option("--max-k", serve.server.limits.max_k, "Largest k")
      ->capture_default_str()
      ->check(CLI::Range(1U, vf::SearchParams::kMaxK));
  serve_cmd->add_option("--max-ef", serve.server.limits.max_ef, "Largest ef_search")
      ->capture_default_str()
      ->check(CLI::Range(1U, vf::HnswParams::kMaxEf));
  serve_cmd->add_option("--max-dim", serve.server.limits.max_dim, "Largest collection dimension")
      ->capture_default_str()
      ->check(CLI::Range(1U, vf::kMaxDim));
  serve_cmd
      ->add_option("--drain-seconds", serve_drain_seconds,
                   "Shutdown waits this long for requests in progress")
      ->capture_default_str();
  serve_cmd->add_flag("--no-mmap", serve_no_mmap, "Load snapshots into memory");
  serve_cmd->add_flag("--snapshot-on-exit", serve.snapshot_on_exit,
                      "Snapshot every collection after a graceful shutdown");
  serve_cmd->add_flag("--list-routes", serve_list_routes, "Print the route table and exit");
#endif

  CLI11_PARSE(app, argc, argv);

  // A VF_SIMD request this build or CPU cannot honour is an error for every command.
  if (const vf::Status simd = vf::simd_status(); !simd.ok()) {
    return report(simd);
  }

#if defined(VF_HAVE_SERVER)
  if (serve_cmd->parsed()) {
    if (serve_list_routes) {
      for (const vf::server::RouteInfo& route : vf::server::Server::routes()) {
        std::cout << route.method << ' ' << route.path << '\n';
      }
      return 0;
    }
    if (serve.data_dir.empty()) {
      std::cerr << "error: --data-dir is required\n";
      return 2;
    }
    if (serve.server.api_key.empty()) {
      serve.server.api_key = environment("VF_API_KEY");
    }
    serve.server.limits.max_body_bytes = serve_max_body_mb << 20U;
    serve.server.drain_timeout = std::chrono::seconds(serve_drain_seconds);
    serve.use_mmap = !serve_no_mmap;
    return report(vf::cli::run_serve(serve, std::cout));
  }
#endif
  if (build_cmd->parsed()) {
    const vf::Result<vf::Metric> metric = vf::parse_metric(build_metric);
    const vf::Result<vf::IndexType> index = vf::parse_index_type(build_index);
    if (!metric.ok() || !index.ok()) {
      return report(!metric.ok() ? metric.status() : index.status());
    }
    build.config.metric = metric.value();
    build.config.index = index.value();
    return report(vf::cli::run_build(build, std::cout));
  }
  if (search_cmd->parsed()) {
    if (search_ef != 0) {
      search.ef_search = search_ef;
    }
    search.use_mmap = !search_no_mmap;
    return report(vf::cli::run_search(search, std::cout));
  }
  if (info_cmd->parsed()) {
    return report(vf::cli::run_info(info_path, std::cout));
  }
  if (verify_cmd->parsed()) {
    return report(vf::cli::run_verify(verify_path, std::cout));
  }

  if (gen_cmd->parsed()) {
    gen.spec.distribution = gen_distribution == "uniform"
                                ? vf::detail::SyntheticDistribution::Uniform
                                : vf::detail::SyntheticDistribution::GaussianMixture;
    gen.format =
        gen_format == "fvecs" ? vf::cli::DatasetFormat::Fvecs : vf::cli::DatasetFormat::Npy;
    return report(vf::cli::run_gen_data(gen, std::cout));
  }
  if (gt_cmd->parsed()) {
    const vf::Result<vf::Metric> metric = vf::parse_metric(gt_metric);
    if (!metric.ok()) {
      return report(metric.status());
    }
    gt.metric = metric.value();
    return report(vf::cli::run_ground_truth(gt, std::cout));
  }
  return 1;
}

}  // namespace
