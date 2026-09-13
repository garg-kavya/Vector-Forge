// Tombstone semantics of HNSW collections (docs/DESIGN.md §9.9) and a model-based random operation
// test: an HNSW Collection and a trivially correct model receive the same seeded sequence; results
// must contain only live ids, be sorted, have correct distances, and meet a recall floor
// (§15.2: "For HNSW, the model checks membership and a recall floor").

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include <vectorforge/collection.hpp>

#include "core/rng.hpp"
#include "support/brute_force_reference.hpp"
#include "support/hnsw_fixture.hpp"
#include "support/test_data.hpp"

namespace {

using vf::Collection;
using vf::ErrorCode;
using vf::ExternalId;
using vf::InternalId;
using vf::Metric;
using vf::Neighbor;

TEST(HnswTombstones, DeletedNodesNeverReturnedAndGraphUnchanged) {
  constexpr std::uint32_t kDim = 16;
  constexpr std::size_t kRows = 3000;
  vf::HnswParams hp;
  hp.M = 12;
  hp.ef_construction = 100;
  vf::test::HnswFixture f(kDim, Metric::L2, hp);
  const vf::test::ClusteredData data(77, kDim, 20, 0.2F);
  const std::vector<float> rows = data.rows(1, kRows);
  f.add_rows(rows);
  const std::vector<std::uint8_t> before = f.backend->graph().canonical_bytes();

  vf::detail::Xoshiro256ss rng(123);
  std::set<InternalId> removed;
  while (removed.size() < kRows * 3 / 10) {
    const auto id = static_cast<InternalId>(vf::detail::uniform_below(rng, kRows));
    if (removed.insert(id).second) {
      f.remove(id);
    }
  }
  EXPECT_EQ(f.backend->graph().canonical_bytes(), before)
      << "removal only sets tombstones; nodes stay as navigation hubs";

  std::vector<vf::test::RefItem> live;
  for (std::size_t i = 0; i < kRows; ++i) {
    if (!removed.contains(static_cast<InternalId>(i))) {
      live.push_back({i, i, std::span<const float>(rows).subspan(i * kDim, kDim)});
    }
  }
  const std::vector<float> queries = data.rows(2, 100);
  std::size_t hits = 0;
  for (std::size_t q = 0; q < 100; ++q) {
    const auto query = std::span<const float>(queries).subspan(q * kDim, kDim);
    const std::vector<Neighbor> result = f.search(query, 10, 128);
    ASSERT_EQ(result.size(), 10U);
    const auto ranking = vf::test::reference_ranking(Metric::L2, false, query, live);
    std::set<ExternalId> exact;
    for (std::size_t i = 0; i < 10; ++i) {
      exact.insert(ranking[i].id);
    }
    for (const Neighbor& n : result) {
      ASSERT_FALSE(removed.contains(static_cast<InternalId>(n.id))) << "deleted id " << n.id;
      hits += exact.contains(n.id) ? 1U : 0U;
    }
  }
  const double recall = static_cast<double>(hits) / 1000.0;
  std::printf("[ measured ] recall@10 over live vectors with 30%% deleted: %.4f\n", recall);
  EXPECT_GE(recall, 0.95);
}

struct ModelEntry {
  std::vector<float> raw;
  std::uint64_t order = 0;
};

class HnswModelTest : public testing::TestWithParam<Metric> {};

TEST_P(HnswModelTest, RandomOperationsKeepMembershipAndRecall) {
  const Metric metric = GetParam();
  constexpr std::uint32_t kDim = 12;
  constexpr int kOperations = 10000;
  constexpr std::uint64_t kIdSpace = 600;
  const bool normalized = metric == Metric::Cosine;

  vf::CollectionConfig cfg;
  cfg.dim = kDim;
  cfg.metric = metric;
  cfg.index = vf::IndexType::Hnsw;
  cfg.hnsw.M = 8;
  cfg.hnsw.ef_construction = 64;
  const std::unique_ptr<Collection> col = Collection::create(cfg).value();

  std::map<ExternalId, ModelEntry> model;
  std::uint64_t rows = 0;
  vf::detail::Xoshiro256ss rng(0xBEEF + static_cast<std::uint64_t>(metric));
  std::size_t searched = 0;
  std::size_t found = 0;

  for (int op = 0; op < kOperations; ++op) {
    SCOPED_TRACE(testing::Message() << "operation " << op);
    const ExternalId id = vf::detail::uniform_below(rng, kIdSpace);
    const std::uint64_t choice = vf::detail::uniform_below(rng, 100);
    const bool exists = model.contains(id);
    if (choice < 45) {
      std::vector<float> v = vf::test::random_vector(rng, kDim);
      const bool upsert = choice >= 35;
      const vf::Status st = col->add(id, v, {.upsert = upsert});
      if (exists && !upsert) {
        ASSERT_EQ(st.code(), ErrorCode::AlreadyExists);
      } else {
        ASSERT_TRUE(st.ok()) << st.to_string();
        model[id] = {std::move(v), rows++};
      }
    } else if (choice < 65) {
      const vf::Status st = col->remove(id);
      ASSERT_EQ(st.ok(), exists);
      model.erase(id);
    } else {
      const std::vector<float> query = vf::test::random_vector(rng, kDim);
      const std::uint32_t k = 1 + static_cast<std::uint32_t>(vf::detail::uniform_below(rng, 20));
      vf::SearchParams params;
      params.k = k;
      params.ef_search = 64;
      const auto result = col->search(query, params);
      ASSERT_TRUE(result.ok()) << result.status().to_string();
      const std::vector<Neighbor>& hits = result.value();

      std::vector<vf::test::RefItem> items;
      for (const auto& [mid, entry] : model) {
        items.push_back({mid, entry.order, entry.raw});
      }
      const auto ranking = vf::test::reference_ranking(metric, normalized, query, items);
      std::unordered_map<ExternalId, double> ref_distance;
      for (const auto& r : ranking) {
        ref_distance.emplace(r.id, r.distance);
      }
      const std::size_t expected = std::min<std::size_t>(k, model.size());
      ASSERT_LE(hits.size(), expected);
      std::set<ExternalId> seen;
      for (std::size_t i = 0; i < hits.size(); ++i) {
        const auto it = ref_distance.find(hits[i].id);
        ASSERT_NE(it, ref_distance.end()) << "result id " << hits[i].id << " is not live";
        ASSERT_NEAR(static_cast<double>(hits[i].distance), it->second, 1e-4);
        ASSERT_TRUE(seen.insert(hits[i].id).second) << "duplicate id";
        if (i > 0) {
          ASSERT_LE(hits[i - 1].distance, hits[i].distance);
        }
      }
      if (expected > 0) {
        const double kth = ranking[expected - 1].distance;
        for (const Neighbor& n : hits) {
          found += ref_distance[n.id] <= kth + 1e-5 ? 1U : 0U;
        }
        searched += expected;
      }
    }
    ASSERT_EQ(col->size(), model.size());
  }
  const double recall = static_cast<double>(found) / static_cast<double>(searched);
  std::printf("[ measured ] metric=%s model recall=%.4f over %zu expected results\n",
              std::string(vf::to_string(metric)).c_str(), recall, searched);
  EXPECT_GE(recall, 0.97) << "recall over " << searched << " expected results";
  const vf::CollectionStats st = col->stats();
  EXPECT_EQ(st.live_count, model.size());
  EXPECT_EQ(st.row_count, rows);
}

INSTANTIATE_TEST_SUITE_P(Metrics, HnswModelTest,
                         testing::Values(Metric::L2, Metric::InnerProduct, Metric::Cosine),
                         [](const testing::TestParamInfo<Metric>& param_info) {
                           return std::string(vf::to_string(param_info.param));
                         });

}  // namespace
