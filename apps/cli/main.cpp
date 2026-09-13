// vectorforge command-line interface.

#include <cstdint>
#include <exception>
#include <iostream>
#include <map>
#include <new>
#include <string>

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

  CLI11_PARSE(app, argc, argv);

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
