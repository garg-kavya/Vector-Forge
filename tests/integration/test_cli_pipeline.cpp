// End-to-end: gen-data -> ground-truth in-process, validated against an independent brute force.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
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

// SIFT-style pipeline (docs/DESIGN.md §21 Phase 4 acceptance): fvecs base/queries, ground truth as
// .ivecs ids and as npy, then build -> info -> verify -> search in both load modes.
TEST(CliPipeline, BuildInfoVerifySearchOnSiftFormatFiles) {
  const vf::test::ScopedTempDir dir("cli_index");
  std::ostringstream log;
  vf::cli::GenDataOptions base;
  base.rows = 4000;
  base.spec.dim = 32;
  base.spec.clusters = 30;
  base.spec.spread = 0.15F;
  base.spec.seed = 11;
  base.format = vf::cli::DatasetFormat::Fvecs;
  base.out = dir.file("sift_base.fvecs");
  ASSERT_TRUE(vf::cli::run_gen_data(base, log).ok());
  vf::cli::GenDataOptions queries = base;
  queries.rows = 60;
  queries.spec.seed = 12;
  queries.out = dir.file("sift_query.fvecs");
  ASSERT_TRUE(vf::cli::run_gen_data(queries, log).ok());

  vf::cli::GroundTruthOptions gt;
  gt.base = base.out;
  gt.queries = queries.out;
  gt.k = 10;
  gt.out_prefix = dir.file("gt");
  ASSERT_TRUE(vf::cli::run_ground_truth(gt, log).ok());
  // Also as TEXMEX .ivecs (the format SIFT ships its ground truth in).
  const auto gt_ids =
      vf::detail::read_npy<std::int64_t>(vf::cli::ground_truth_ids_path(gt.out_prefix)).value();
  std::vector<std::int32_t> ids32;
  for (const std::int64_t id : gt_ids.data) {
    ids32.push_back(static_cast<std::int32_t>(id));
  }
  ASSERT_TRUE(
      vf::detail::write_ivecs(dir.file("sift_groundtruth.ivecs"), ids32, gt_ids.rows, gt_ids.cols)
          .ok());

  for (const vf::IndexType index : {vf::IndexType::Hnsw, vf::IndexType::Flat}) {
    SCOPED_TRACE(vf::to_string(index));
    vf::cli::BuildOptions build;
    build.input = base.out;
    build.config.index = index;
    build.config.hnsw.M = 12;
    build.config.hnsw.ef_construction = 100;
    build.out = dir.file(std::string(vf::to_string(index)) + ".vfidx");
    vf::Status st = vf::cli::run_build(build, log);
    ASSERT_TRUE(st.ok()) << st.to_string();

    std::ostringstream info;
    st = vf::cli::run_info(build.out, info);
    ASSERT_TRUE(st.ok()) << st.to_string();
    EXPECT_NE(info.str().find("rows 4000"), std::string::npos) << info.str();
    EXPECT_NE(info.str().find("VECTORS"), std::string::npos);
    std::ostringstream verify;
    st = vf::cli::run_verify(build.out, verify);
    ASSERT_TRUE(st.ok()) << st.to_string();
    EXPECT_NE(verify.str().find("OK"), std::string::npos);

    for (const bool mmap : {true, false}) {
      for (const bool ivecs : {false, true}) {
        vf::cli::SearchOptions search;
        search.index = build.out;
        search.queries = queries.out;
        search.k = 10;
        search.ef_search = 128;
        search.use_mmap = mmap;
        search.ground_truth = ivecs ? dir.file("sift_groundtruth.ivecs") : gt.out_prefix;
        search.out_prefix = dir.file("results");
        std::ostringstream out;
        st = vf::cli::run_search(search, out);
        ASSERT_TRUE(st.ok()) << st.to_string();
        // Recall is printed as "recall@10 <value>"; HNSW at ef 128 on 4000 points is (near) exact.
        const std::string text = out.str();
        const std::size_t at = text.find("recall@10 ");
        ASSERT_NE(at, std::string::npos) << text;
        EXPECT_GE(std::stod(text.substr(at + 10)), 0.99) << text;
        const auto result_ids =
            vf::detail::read_npy<std::int64_t>(vf::cli::ground_truth_ids_path(search.out_prefix))
                .value();
        EXPECT_EQ(result_ids.rows, 60U);
        EXPECT_EQ(result_ids.cols, 10U);
      }
    }
  }

  // Damaged index: verify fails and reports it.
  std::vector<char> bytes(
      static_cast<std::size_t>(std::filesystem::file_size(dir.file("hnsw.vfidx"))));
  {
    std::ifstream in(dir.file("hnsw.vfidx"), std::ios::binary);
    in.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  bytes[bytes.size() / 2] = static_cast<char>(bytes[bytes.size() / 2] ^ 0x10);
  {
    std::ofstream out(dir.file("damaged.vfidx"), std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  std::ostringstream verify;
  EXPECT_EQ(vf::cli::run_verify(dir.file("damaged.vfidx"), verify).code(),
            vf::ErrorCode::CorruptData);
  EXPECT_NE(verify.str().find("FAILED"), std::string::npos);
}

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
