// End-to-end: gen-data -> ground-truth in-process, validated against an independent brute force.

#include <gtest/gtest.h>

#include <cstdint>
#include <sstream>
#include <vector>

#include <vectorforge/types.hpp>

#include "commands.hpp"
#include "support/brute_force_reference.hpp"
#include "support/test_data.hpp"
#include "util/dataset_io.hpp"
#include "util/recall.hpp"

namespace {

using vf::Metric;

struct PipelineCase {
  Metric metric;
  vf::cli::DatasetFormat format;
};

class CliPipelineTest : public testing::TestWithParam<PipelineCase> {};

TEST_P(CliPipelineTest, GroundTruthMatchesIndependentBruteForce) {
  const PipelineCase pc = GetParam();
  const vf::test::ScopedTempDir dir("cli_pipeline");
  const char* ext = pc.format == vf::cli::DatasetFormat::Npy ? ".npy" : ".fvecs";
  std::ostringstream log;

  vf::cli::GenDataOptions base;
  base.rows = 3000;
  base.spec.dim = 24;
  base.spec.clusters = 20;
  base.spec.spread = 0.2F;
  base.spec.seed = 1;
  base.format = pc.format;
  base.out = dir.file(std::string("base") + ext);
  ASSERT_TRUE(vf::cli::run_gen_data(base, log).ok());

  vf::cli::GenDataOptions queries = base;
  queries.rows = 40;
  queries.spec.seed = 2;
  queries.out = dir.file(std::string("queries") + ext);
  ASSERT_TRUE(vf::cli::run_gen_data(queries, log).ok());

  vf::cli::GroundTruthOptions gt;
  gt.base = base.out;
  gt.queries = queries.out;
  gt.metric = pc.metric;
  gt.k = 25;
  gt.out_prefix = dir.file("gt");
  const vf::Status st = vf::cli::run_ground_truth(gt, log);
  ASSERT_TRUE(st.ok()) << st.to_string();

  const auto ids =
      vf::detail::read_npy<std::int64_t>(vf::cli::ground_truth_ids_path(gt.out_prefix));
  const auto dists =
      vf::detail::read_npy<float>(vf::cli::ground_truth_distances_path(gt.out_prefix));
  ASSERT_TRUE(ids.ok()) << ids.status().to_string();
  ASSERT_TRUE(dists.ok()) << dists.status().to_string();
  ASSERT_EQ(ids.value().rows, 40U);
  ASSERT_EQ(ids.value().cols, 25U);
  ASSERT_EQ(dists.value().cols, 25U);

  const auto base_m = vf::detail::read_float_matrix(base.out).value();
  const auto query_m = vf::detail::read_float_matrix(queries.out).value();
  ASSERT_EQ(base_m.rows, 3000U);
  std::vector<vf::test::RefItem> items;
  for (std::uint64_t i = 0; i < base_m.rows; ++i) {
    items.push_back({i, i, base_m.row(i)});
  }
  const bool normalized = pc.metric == Metric::Cosine;
  std::vector<vf::ExternalId> gt_ids;
  for (std::uint64_t q = 0; q < query_m.rows; ++q) {
    std::vector<vf::Neighbor> hits;
    for (std::uint64_t j = 0; j < 25; ++j) {
      const std::int64_t id = ids.value().data[(q * 25) + j];
      ASSERT_GE(id, 0);
      hits.push_back({static_cast<vf::ExternalId>(id), dists.value().data[(q * 25) + j]});
      gt_ids.push_back(static_cast<vf::ExternalId>(id));
    }
    const auto ranking = vf::test::reference_ranking(pc.metric, normalized, query_m.row(q), items);
    const double tol = 1e-4 * std::max(1.0, std::fabs(ranking.front().distance));
    vf::test::expect_tolerant_topk(hits, ranking, 25, tol);
  }

  // Ground truth has perfect recall against itself.
  const auto recall = vf::detail::recall_at_k(gt_ids, dists.value().data, 25, gt_ids,
                                              dists.value().data, 25, 40, 10);
  ASSERT_TRUE(recall.ok());
  EXPECT_DOUBLE_EQ(recall.value().mean, 1.0);
}

INSTANTIATE_TEST_SUITE_P(
    Metrics, CliPipelineTest,
    testing::Values(PipelineCase{Metric::L2, vf::cli::DatasetFormat::Npy},
                    PipelineCase{Metric::InnerProduct, vf::cli::DatasetFormat::Npy},
                    PipelineCase{Metric::Cosine, vf::cli::DatasetFormat::Fvecs}),
    [](const testing::TestParamInfo<PipelineCase>& param_info) {
      return std::string(vf::to_string(param_info.param.metric)) +
             (param_info.param.format == vf::cli::DatasetFormat::Npy ? "_npy" : "_fvecs");
    });

TEST(CliCommands, ReportErrors) {
  std::ostringstream log;
  vf::cli::GenDataOptions gen;
  gen.spec.dim = 4;
  EXPECT_EQ(vf::cli::run_gen_data(gen, log).code(), vf::ErrorCode::InvalidArgument);  // rows 0
  gen.rows = 5;
  EXPECT_EQ(vf::cli::run_gen_data(gen, log).code(), vf::ErrorCode::InvalidArgument);  // no out

  const vf::test::ScopedTempDir dir("cli_errors");
  gen.out = dir.file("a.npy");
  ASSERT_TRUE(vf::cli::run_gen_data(gen, log).ok());
  vf::cli::GenDataOptions other = gen;
  other.spec.dim = 5;
  other.out = dir.file("b.npy");
  ASSERT_TRUE(vf::cli::run_gen_data(other, log).ok());

  vf::cli::GroundTruthOptions gt;
  gt.base = gen.out;
  gt.queries = other.out;
  gt.out_prefix = dir.file("gt");
  EXPECT_EQ(vf::cli::run_ground_truth(gt, log).code(), vf::ErrorCode::DimensionMismatch);
  gt.queries = dir.file("missing.npy");
  EXPECT_EQ(vf::cli::run_ground_truth(gt, log).code(), vf::ErrorCode::IoError);
}

}  // namespace
