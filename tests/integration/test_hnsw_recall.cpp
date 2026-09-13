// HNSW recall against exact ground truth (docs/DESIGN.md §15.3): N = 10 000 clustered vectors,
// d in {16, 128}, metrics {L2, IP, cosine}, fixed seeds, recall@10 at ef_search in {32, 128}.
// Every build is also checked by HnswValidator (invariants + full level-0 reachability).
//
// Thresholds are regression guards frozen from measured values minus a margin (see kCases); they
// are not performance claims. Measured recall is printed for every case.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <vectorforge/collection.hpp>

#include "support/hnsw_fixture.hpp"
#include "support/test_data.hpp"
#include "util/recall.hpp"

namespace {

using vf::ExternalId;
using vf::Metric;
using vf::Neighbor;

constexpr std::size_t kRows = 10000;
constexpr std::size_t kQueries = 200;
constexpr std::uint32_t kK = 10;

struct RecallCase {
  Metric metric;
  std::uint32_t dim;
  double min_recall_ef32;
  double min_recall_ef128;
  std::uint64_t max_unreachable;  // per level
};

class HnswRecallTest : public testing::TestWithParam<RecallCase> {};

TEST_P(HnswRecallTest, MeetsFrozenThresholds) {
  const RecallCase rc = GetParam();
  const vf::test::ClusteredData data(1000 + rc.dim, rc.dim, 50, 0.25F);
  const std::vector<float> base = data.rows(1, kRows);
  const std::vector<float> queries = data.rows(2, kQueries);

  // Exact ground truth from a Flat collection (its own oracle tests are in test_flat_backend).
  vf::CollectionConfig flat_cfg;
  flat_cfg.dim = rc.dim;
  flat_cfg.metric = rc.metric;
  flat_cfg.index = vf::IndexType::Flat;
  const std::unique_ptr<vf::Collection> flat = vf::Collection::create(flat_cfg).value();
  std::vector<ExternalId> ids(kRows);
  for (std::size_t i = 0; i < kRows; ++i) {
    ids[i] = i;
  }
  ASSERT_EQ(flat->add_batch(ids, base).value(), kRows);
  vf::SearchParams sp;
  sp.k = kK;
  std::vector<ExternalId> gt_ids(kQueries * kK);
  std::vector<float> gt_dist(kQueries * kK);
  std::vector<std::uint32_t> gt_counts(kQueries);
  ASSERT_TRUE(flat->search_batch(queries, kQueries, sp, gt_ids, gt_dist, gt_counts).ok());

  vf::test::HnswFixture f(rc.dim, rc.metric, vf::HnswParams{});
  f.add_rows(base);
  const vf::Status invariants = f.validator().check_invariants();
  ASSERT_TRUE(invariants.ok()) << invariants.to_string();
  const vf::detail::HnswReachability reach = f.validator().reachability();
  for (std::size_t l = 0; l < reach.unreachable.size(); ++l) {
    std::printf("[ measured ] metric=%s dim=%u level=%zu unreachable=%llu\n",
                std::string(vf::to_string(rc.metric)).c_str(), rc.dim, l,
                static_cast<unsigned long long>(reach.unreachable[l]));
    EXPECT_LE(reach.unreachable[l], rc.max_unreachable) << "level " << l;
  }

  double previous = 0.0;
  for (const std::uint32_t ef : {16U, 32U, 64U, 128U}) {
    std::vector<ExternalId> approx_ids(kQueries * kK, vf::kInvalidExternalId);
    std::vector<float> approx_dist(kQueries * kK, std::numeric_limits<float>::infinity());
    for (std::size_t q = 0; q < kQueries; ++q) {
      const std::vector<Neighbor> hits =
          f.search(std::span<const float>(queries).subspan(q * rc.dim, rc.dim), kK, ef);
      ASSERT_EQ(hits.size(), kK);
      for (std::size_t j = 0; j < hits.size(); ++j) {
        approx_ids[(q * kK) + j] = hits[j].id;
        approx_dist[(q * kK) + j] = hits[j].distance;
      }
    }
    const vf::detail::RecallStats recall =
        vf::detail::recall_at_k(approx_ids, approx_dist, kK, gt_ids, gt_dist, kK, kQueries, kK)
            .value();
    std::printf("[ measured ] metric=%s dim=%u ef=%u recall@10 mean=%.4f min=%.2f\n",
                std::string(vf::to_string(rc.metric)).c_str(), rc.dim, ef, recall.mean, recall.min);
    EXPECT_GE(recall.mean, previous) << "recall must not decrease as ef grows (ef=" << ef << ")";
    previous = recall.mean;
    if (ef == 32) {
      EXPECT_GE(recall.mean, rc.min_recall_ef32);
    } else if (ef == 128) {
      EXPECT_GE(recall.mean, rc.min_recall_ef128);
    }
  }
}

// Thresholds (docs/hnsw.md, "Test thresholds"): recall = measured mean minus 0.02 at ef = 32 and
// minus 0.01 at ef = 128, rounded down to 0.01. L2 and cosine graphs must be fully reachable.
// Inner product is not a metric: pruning evicts low-norm nodes from every list, so a fraction of
// nodes becomes unreachable; the bound is the measured count plus 20 %.
const RecallCase kCases[] = {
    {Metric::L2, 16, 0.98, 0.99, 0},
    {Metric::InnerProduct, 16, 0.97, 0.99, 603},  // measured 503 unreachable on level 0
    {Metric::Cosine, 16, 0.98, 0.99, 0},
    {Metric::L2, 128, 0.97, 0.99, 0},
    {Metric::InnerProduct, 128, 0.98, 0.99, 267},  // measured 223 unreachable on level 0
    {Metric::Cosine, 128, 0.97, 0.99, 0},
};

INSTANTIATE_TEST_SUITE_P(Configs, HnswRecallTest, testing::ValuesIn(kCases),
                         [](const testing::TestParamInfo<RecallCase>& param_info) {
                           return std::string(vf::to_string(param_info.param.metric)) + "_d" +
                                  std::to_string(param_info.param.dim);
                         });

}  // namespace
