// Level B parallel construction (docs/DESIGN.md §11.7; docs/concurrency.md "Level B"):
//   - linking serially in the Concurrent mode builds exactly the graph the Coarse mode builds;
//   - parallel builds with 1..8 threads pass the graph validator, keep every row reachable, and
//     reach the recall of the serial build within a tolerance;
//   - ids, vectors, upserts and statistics are consistent after a parallel build.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <span>
#include <vector>

#include <vectorforge/collection.hpp>
#include <vectorforge/thread_pool.hpp>

#include "collection/collection_factory.hpp"
#include "index/hnsw/hnsw_backend.hpp"
#include "index/hnsw/hnsw_validator.hpp"
#include "support/test_data.hpp"
#include "util/recall.hpp"

namespace {

using vf::Collection;
using vf::Concurrency;
using vf::ExternalId;

constexpr std::uint32_t kDim = 16;
constexpr std::uint32_t kK = 10;
constexpr std::size_t kQueries = 200;

const vf::detail::HnswBackend& backend_of(const Collection& c) {
  return static_cast<const vf::detail::HnswBackend&>(
      *vf::detail::CollectionFactory::state(c).backend);
}

std::unique_ptr<Collection> make(Concurrency mode, vf::IndexType index = vf::IndexType::Hnsw) {
  vf::CollectionConfig cfg;
  cfg.dim = kDim;
  cfg.index = index;
  cfg.concurrency = mode;
  cfg.hnsw.M = 12;
  cfg.hnsw.ef_construction = 100;
  return Collection::create(cfg).value();
}

std::vector<ExternalId> iota_ids(std::size_t n, ExternalId base = 0) {
  std::vector<ExternalId> ids(n);
  for (std::size_t i = 0; i < n; ++i) {
    ids[i] = base + i;
  }
  return ids;
}

// Mean recall@k of `c` against exact results for `queries`.
double recall_of(const Collection& c, std::span<const float> queries,
                 const std::vector<ExternalId>& gt_ids, const std::vector<float>& gt_dist) {
  vf::SearchParams p;
  p.k = kK;
  p.ef_search = kK;  // small beam: recall stays below 1 and reflects graph quality
  std::vector<ExternalId> ids(kQueries * kK);
  std::vector<float> dist(kQueries * kK);
  std::vector<std::uint32_t> counts(kQueries);
  EXPECT_TRUE(c.search_batch(queries, kQueries, p, ids, dist, counts).ok());
  return vf::detail::recall_at_k(ids, dist, kK, gt_ids, gt_dist, kK, kQueries, kK).value().mean;
}

TEST(ParallelBuild, SerialLinkingMatchesCoarseMode) {
  const vf::test::ClusteredData data(5, kDim, 20, 0.3F);
  const std::vector<float> rows = data.rows(1, 3000);
  const std::vector<ExternalId> ids = iota_ids(3000);
  const std::unique_ptr<Collection> coarse = make(Concurrency::Coarse);
  const std::unique_ptr<Collection> linked = make(Concurrency::Concurrent);
  // Uneven batches cross several grow/link sections.
  for (std::size_t first = 0; first < ids.size(); first += 700) {
    const std::size_t count = std::min<std::size_t>(700, ids.size() - first);
    const auto id_span = std::span<const ExternalId>(ids).subspan(first, count);
    const auto row_span = std::span<const float>(rows).subspan(first * kDim, count * kDim);
    ASSERT_EQ(coarse->add_batch(id_span, row_span).value(), count);
    ASSERT_EQ(linked->add_batch(id_span, row_span).value(), count);
  }
  EXPECT_EQ(backend_of(*coarse).graph().canonical_bytes(),
            backend_of(*linked).graph().canonical_bytes());
  EXPECT_EQ(backend_of(*coarse).build_stats().distance_computations,
            backend_of(*linked).build_stats().distance_computations);
}

TEST(ParallelBuild, ValidAndCloseToSerialRecall) {
  constexpr std::size_t kRows = 8000;
  const vf::test::ClusteredData data(9, kDim, 40, 0.6F);
  const std::vector<float> rows = data.rows(1, kRows);
  const std::vector<float> queries = data.rows(2, kQueries);
  const std::vector<ExternalId> ids = iota_ids(kRows);

  const std::unique_ptr<Collection> flat = make(Concurrency::Coarse, vf::IndexType::Flat);
  ASSERT_EQ(flat->add_batch(ids, rows).value(), kRows);
  vf::SearchParams exact;
  exact.k = kK;
  std::vector<ExternalId> gt_ids(kQueries * kK);
  std::vector<float> gt_dist(kQueries * kK);
  std::vector<std::uint32_t> gt_counts(kQueries);
  ASSERT_TRUE(flat->search_batch(queries, kQueries, exact, gt_ids, gt_dist, gt_counts).ok());

  const std::unique_ptr<Collection> serial = make(Concurrency::Coarse);
  ASSERT_EQ(serial->add_batch(ids, rows).value(), kRows);
  const double serial_recall = recall_of(*serial, queries, gt_ids, gt_dist);
  std::printf("[ measured ] serial recall@10=%.4f\n", serial_recall);

  for (const std::size_t threads : {1U, 2U, 4U, 8U}) {
    vf::ThreadPool pool(threads - 1);
    const std::unique_ptr<Collection> c = make(Concurrency::Concurrent);
    ASSERT_EQ(c->add_batch(ids, rows, {}, threads > 1 ? &pool : nullptr).value(), kRows);
    ASSERT_EQ(c->size(), kRows);

    const vf::detail::HnswGraph& graph = backend_of(*c).graph();
    const vf::detail::HnswValidator validator(graph);
    const vf::Status invariants = validator.check_invariants();
    ASSERT_TRUE(invariants.ok()) << threads << " threads: " << invariants.to_string();
    const vf::detail::HnswReachability reach = validator.reachability();
    EXPECT_EQ(reach.unreachable_level0(), 0U) << threads << " threads";

    const double recall = recall_of(*c, queries, gt_ids, gt_dist);
    std::printf("[ measured ] threads=%zu recall@10=%.4f\n", threads, recall);
    // Tolerance: parallel builds differ from the serial one only in insertion interleaving;
    // measured differences are below 0.005 on this data.
    EXPECT_GE(recall, serial_recall - 0.02) << threads << " threads";
  }
}

TEST(ParallelBuild, IdsVectorsAndUpsertsAreConsistent) {
  constexpr std::size_t kRows = 3000;
  const vf::test::ClusteredData data(3, kDim, 10, 0.3F);
  const std::vector<float> rows = data.rows(1, kRows);
  const std::vector<float> replacement = data.rows(7, kRows / 2);
  const std::vector<ExternalId> ids = iota_ids(kRows, 100);
  vf::ThreadPool pool(3);
  const std::unique_ptr<Collection> c = make(Concurrency::Concurrent);
  ASSERT_EQ(c->add_batch(ids, rows, {}, &pool).value(), kRows);

  // Existing ids are rejected before anything changes.
  EXPECT_EQ(c->add_batch(std::span<const ExternalId>(ids).first(1),
                         std::span<const float>(rows).first(kDim), {}, &pool)
                .status()
                .code(),
            vf::ErrorCode::AlreadyExists);

  // Upsert every other id: old rows become tombstones.
  std::vector<ExternalId> odd;
  for (std::size_t i = 1; i < kRows; i += 2) {
    odd.push_back(ids[i]);
  }
  ASSERT_EQ(c->add_batch(odd, replacement, {.upsert = true}, &pool).value(), odd.size());
  const vf::CollectionStats st = c->stats();
  EXPECT_EQ(st.live_count, kRows);
  EXPECT_EQ(st.deleted_count, odd.size());
  EXPECT_EQ(st.row_count, kRows + odd.size());

  vf::SearchParams p;
  p.k = 1;
  p.ef_search = 128;
  std::size_t found = 0;
  for (std::size_t i = 0; i < kRows; ++i) {
    const bool replaced = i % 2 == 1;
    const std::span<const float> expected =
        replaced ? std::span<const float>(replacement).subspan((i / 2) * kDim, kDim)
                 : std::span<const float>(rows).subspan(i * kDim, kDim);
    const std::vector<float> stored = c->get(ids[i]).value();
    ASSERT_TRUE(std::equal(stored.begin(), stored.end(), expected.begin())) << "id " << ids[i];
    const std::vector<vf::Neighbor> hits = c->search(expected, p).value();
    if (!hits.empty() && hits[0].distance == 0.0F) {
      ++found;
    }
  }
  EXPECT_GE(found, kRows * 99 / 100);

  const vf::detail::HnswValidator validator(backend_of(*c).graph());
  EXPECT_TRUE(validator.check_invariants().ok());
  EXPECT_GT(backend_of(*c).build_stats().distance_computations, 0U);
}

TEST(ParallelBuild, TinyBatchesAndEmptyCollection) {
  vf::ThreadPool pool(4);
  const std::unique_ptr<Collection> c = make(Concurrency::Concurrent);
  const vf::test::ClusteredData data(4, kDim, 3, 0.3F);
  // First batch into an empty graph: several threads race for the entry point.
  const std::vector<float> rows = data.rows(1, 40);
  for (std::size_t i = 0; i < 40; i += 8) {
    const std::vector<ExternalId> ids = iota_ids(8, i);
    ASSERT_EQ(c->add_batch(ids, std::span<const float>(rows).subspan(i * kDim, 8 * kDim), {}, &pool)
                  .value(),
              8U);
  }
  const vf::detail::HnswValidator validator(backend_of(*c).graph());
  ASSERT_TRUE(validator.check_invariants().ok()) << validator.check_invariants().to_string();
  EXPECT_EQ(validator.reachability().unreachable_level0(), 0U);
  EXPECT_EQ(c->size(), 40U);
}

}  // namespace
