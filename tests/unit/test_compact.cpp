// Collection::compact() and the parallel batch paths (docs/concurrency.md).

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <vectorforge/collection.hpp>
#include <vectorforge/thread_pool.hpp>

#include "collection/collection_factory.hpp"
#include "index/hnsw/hnsw_backend.hpp"
#include "index/hnsw/hnsw_validator.hpp"
#include "support/test_data.hpp"

namespace {

using vf::Collection;
using vf::ExternalId;
using vf::IndexType;
using vf::Neighbor;

constexpr std::uint32_t kDim = 10;
constexpr std::size_t kRows = 600;

std::unique_ptr<Collection> make(IndexType index, vf::Metric metric = vf::Metric::Cosine) {
  vf::CollectionConfig cfg;
  cfg.dim = kDim;
  cfg.metric = metric;
  cfg.index = index;
  cfg.hnsw.M = 8;
  cfg.hnsw.ef_construction = 64;
  return Collection::create(cfg).value();
}

std::vector<ExternalId> ids_from(ExternalId first, std::size_t n) {
  std::vector<ExternalId> ids(n);
  for (std::size_t i = 0; i < n; ++i) {
    ids[i] = first + (3 * i);
  }
  return ids;
}

class Compact : public testing::TestWithParam<IndexType> {};

TEST_P(Compact, DropsRemovedRowsAndKeepsEverythingElse) {
  const auto c = make(GetParam());
  const vf::test::ClusteredData data(4, kDim, 10, 0.2F);
  const std::vector<ExternalId> ids = ids_from(100, kRows);
  ASSERT_EQ(c->add_batch(ids, data.rows(1, kRows)).value(), kRows);
  // Remove a third, upsert a few (upsert leaves a tombstone too).
  for (std::size_t i = 0; i < kRows; i += 3) {
    ASSERT_TRUE(c->remove(ids[i]).ok());
  }
  const std::vector<float> replacement = data.rows(2, 5);
  for (std::size_t i = 0; i < 5; ++i) {
    ASSERT_TRUE(c->add(ids[1 + (3 * i)],
                       std::span<const float>(replacement).subspan(i * kDim, kDim),
                       {.upsert = true})
                    .ok());
  }
  const std::size_t live = c->size();
  std::vector<std::pair<ExternalId, std::vector<float>>> stored;
  for (const ExternalId id : ids) {
    if (c->contains(id)) {
      stored.emplace_back(id, c->get(id).value());
    }
  }
  const vf::CollectionStats before = c->stats();
  ASSERT_EQ(before.deleted_count, 205U);

  const auto result = c->compact();
  ASSERT_TRUE(result.ok()) << result.status().to_string();
  EXPECT_EQ(result.value().rows_before, before.row_count);
  EXPECT_EQ(result.value().removed_rows, 205U);
  EXPECT_EQ(result.value().rows_after, live);

  const vf::CollectionStats after = c->stats();
  EXPECT_EQ(after.deleted_count, 0U);
  EXPECT_EQ(after.row_count, live);
  EXPECT_EQ(after.live_count, live);
  EXPECT_EQ(c->size(), live);
  for (const auto& [id, v] : stored) {
    ASSERT_EQ(c->get(id).value(), v) << id;  // bit-identical: rows are not re-normalised
  }
  for (std::size_t i = 0; i < kRows; i += 3) {
    EXPECT_FALSE(c->contains(ids[i]));
  }

  // Search still works and never returns removed ids; Flat results equal a fresh collection's.
  const auto fresh = make(GetParam());
  std::vector<ExternalId> fresh_ids;
  std::vector<float> fresh_rows;
  for (const auto& [id, v] : stored) {
    fresh_ids.push_back(id);
    fresh_rows.insert(fresh_rows.end(), v.begin(), v.end());
  }
  ASSERT_TRUE(fresh->add_batch(fresh_ids, fresh_rows).ok());
  const std::vector<float> queries = data.rows(3, 20);
  vf::SearchParams params;
  params.k = 10;
  params.ef_search = 128;
  for (std::size_t q = 0; q < 20; ++q) {
    const auto query = std::span<const float>(queries).subspan(q * kDim, kDim);
    const std::vector<Neighbor> got = c->search(query, params).value();
    ASSERT_EQ(got.size(), 10U);
    for (const Neighbor& n : got) {
      EXPECT_TRUE(c->contains(n.id));
    }
    if (GetParam() == IndexType::Flat) {
      // Stored rows are already unit length; normalising them again may change the last bit, so
      // compare ids and distances with a tolerance.
      const std::vector<Neighbor> want = fresh->search(query, params).value();
      ASSERT_EQ(got.size(), want.size());
      for (std::size_t j = 0; j < got.size(); ++j) {
        EXPECT_NEAR(got[j].distance, want[j].distance, 1e-5F);
      }
    }
  }
  if (GetParam() == IndexType::Hnsw) {
    const auto& backend = static_cast<const vf::detail::HnswBackend&>(
        *vf::detail::CollectionFactory::state(*c).backend);
    const vf::detail::HnswValidator validator(backend.graph());
    EXPECT_TRUE(validator.check_invariants().ok());
    EXPECT_EQ(backend.graph().node_count(), live);
  }

  // The compacted collection keeps working and round-trips through a file.
  ASSERT_TRUE(c->add(99999, std::span<const float>(replacement).first(kDim)).ok());
  const vf::test::ScopedTempDir dir("compact");
  ASSERT_TRUE(c->save(dir.path() / "c.vfidx").ok());
  const auto loaded = Collection::load(dir.path() / "c.vfidx").value();
  EXPECT_EQ(loaded->size(), live + 1);
  EXPECT_EQ(loaded->stats().deleted_count, 0U);
}

TEST_P(Compact, NoopWithoutRemovalsAndOnEmptyCollections) {
  const auto c = make(GetParam());
  auto empty = c->compact();
  ASSERT_TRUE(empty.ok());
  EXPECT_EQ(empty.value().rows_before, 0U);
  EXPECT_EQ(empty.value().rows_after, 0U);

  const vf::test::ClusteredData data(5, kDim, 4, 0.2F);
  ASSERT_TRUE(c->add_batch(ids_from(0, 50), data.rows(1, 50)).ok());
  const auto nothing = c->compact();
  ASSERT_TRUE(nothing.ok());
  EXPECT_EQ(nothing.value().removed_rows, 0U);
  EXPECT_EQ(nothing.value().rows_after, 50U);

  // Removing everything compacts to an empty, usable collection.
  for (const ExternalId id : ids_from(0, 50)) {
    ASSERT_TRUE(c->remove(id).ok());
  }
  const auto all = c->compact();
  ASSERT_TRUE(all.ok());
  EXPECT_EQ(all.value().rows_after, 0U);
  EXPECT_TRUE(c->search(data.rows(2, 1)).value().empty());
  ASSERT_TRUE(c->add(7, data.rows(3, 1)).ok());
  EXPECT_EQ(c->search(data.rows(2, 1)).value().size(), 1U);
}

TEST_P(Compact, MappedCollectionMovesToHeap) {
  const vf::test::ScopedTempDir dir("compact");
  const std::filesystem::path file = dir.path() / "m.vfidx";
  {
    vf::CollectionConfig cfg;
    cfg.dim = 4096;  // one 16 MiB chunk holds 1024 rows: 1100 rows map a full chunk
    cfg.index = GetParam();
    cfg.hnsw.M = 4;
    cfg.hnsw.ef_construction = 8;
    const auto c = Collection::create(cfg).value();
    const vf::test::ClusteredData data(6, 4096, 3, 0.5F);
    ASSERT_TRUE(c->add_batch(ids_from(0, 1100), data.rows(1, 1100)).ok());
    ASSERT_TRUE(c->save(file).ok());
  }
  auto c = Collection::load(file, {.use_mmap = true}).value();
  ASSERT_GT(c->stats().memory.mapped_vectors_bytes, 0U);
  ASSERT_TRUE(c->remove(0).ok());
  ASSERT_TRUE(c->compact().ok());
  EXPECT_EQ(c->stats().memory.mapped_vectors_bytes, 0U);
  // The mapping was released: the file can be replaced now, even on Windows.
  EXPECT_TRUE(c->save(file).ok());
  EXPECT_EQ(Collection::load(file).value()->size(), 1099U);
}

INSTANTIATE_TEST_SUITE_P(Indexes, Compact, testing::Values(IndexType::Flat, IndexType::Hnsw),
                         [](const testing::TestParamInfo<IndexType>& param_info) {
                           return std::string(vf::to_string(param_info.param));
                         });

TEST(ParallelBatches, SameResultsAsSerial) {
  vf::ThreadPool pool(4);
  for (const IndexType index : {IndexType::Flat, IndexType::Hnsw}) {
    SCOPED_TRACE(std::string(vf::to_string(index)));
    const vf::test::ClusteredData data(8, kDim, 10, 0.2F);
    const std::vector<float> rows = data.rows(1, kRows);
    const auto serial = make(index);
    const auto parallel = make(index);
    ASSERT_TRUE(serial->add_batch(ids_from(0, kRows), rows).ok());
    ASSERT_TRUE(parallel->add_batch(ids_from(0, kRows), rows, {}, &pool).ok());
    for (const ExternalId id : ids_from(0, kRows)) {
      ASSERT_EQ(serial->get(id).value(), parallel->get(id).value());
    }

    constexpr std::size_t kQueries = 257;
    const std::vector<float> queries = data.rows(2, kQueries);
    vf::SearchParams params;
    params.k = 7;
    std::vector<ExternalId> ids_a(kQueries * 7);
    std::vector<ExternalId> ids_b(kQueries * 7);
    std::vector<float> dist_a(kQueries * 7);
    std::vector<float> dist_b(kQueries * 7);
    std::vector<std::uint32_t> counts_a(kQueries);
    std::vector<std::uint32_t> counts_b(kQueries);
    ASSERT_TRUE(serial->search_batch(queries, kQueries, params, ids_a, dist_a, counts_a).ok());
    ASSERT_TRUE(
        serial->search_batch(queries, kQueries, params, ids_b, dist_b, counts_b, &pool).ok());
    EXPECT_EQ(ids_a, ids_b);
    EXPECT_EQ(dist_a, dist_b);
    EXPECT_EQ(counts_a, counts_b);
  }
}

TEST(ParallelBatches, NormalisationErrorsAreReportedBeforeAnyInsert) {
  vf::ThreadPool pool(3);
  const auto c = make(IndexType::Flat);
  std::vector<float> rows(kDim * 1000, 1.0F);
  std::fill_n(rows.begin() + (kDim * 777), kDim, 0.0F);  // row 777 cannot be normalised
  const auto added = c->add_batch(ids_from(0, 1000), rows, {}, &pool);
  ASSERT_FALSE(added.ok());
  EXPECT_EQ(added.status().code(), vf::ErrorCode::InvalidArgument);
  EXPECT_NE(added.status().message().find("row 777"), std::string::npos);
  EXPECT_EQ(c->size(), 0U);
}

}  // namespace
