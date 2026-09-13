// HNSW edge cases (docs/DESIGN.md §15.3) through the public Collection API and, where the graph has
// to be inspected, through HnswBackend directly.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <span>
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
using vf::Metric;
using vf::Neighbor;
using vf::SearchParams;

std::unique_ptr<Collection> make_hnsw(std::uint32_t dim, Metric metric = Metric::L2,
                                      std::uint32_t m = 8) {
  vf::CollectionConfig cfg;
  cfg.dim = dim;
  cfg.metric = metric;
  cfg.index = vf::IndexType::Hnsw;
  cfg.hnsw.M = m;
  cfg.hnsw.ef_construction = 64;
  auto c = Collection::create(cfg);
  EXPECT_TRUE(c.ok()) << c.status().to_string();
  return std::move(c).value();
}

SearchParams params(std::uint32_t k, std::uint32_t ef = 0) {
  SearchParams p;
  p.k = k;
  if (ef != 0) {
    p.ef_search = ef;
  }
  return p;
}

void expect_valid_graph(const vf::test::HnswFixture& f) {
  const vf::Status st = f.validator().check_invariants();
  ASSERT_TRUE(st.ok()) << st.to_string();
  EXPECT_EQ(f.validator().reachability().unreachable_level0(), 0U);
}

TEST(HnswCollection, CreateValidatesHnswParams) {
  vf::CollectionConfig cfg;
  cfg.dim = 4;
  cfg.index = vf::IndexType::Hnsw;
  cfg.hnsw.M = 1;
  EXPECT_EQ(Collection::create(cfg).status().code(), ErrorCode::InvalidArgument);
  cfg.hnsw.M = 16;
  cfg.hnsw.ef_construction = 8;  // < M
  EXPECT_EQ(Collection::create(cfg).status().code(), ErrorCode::InvalidArgument);
  cfg.hnsw.ef_construction = 16;
  const auto ok = Collection::create(cfg);
  ASSERT_TRUE(ok.ok());
  EXPECT_EQ(ok.value()->stats().index, vf::IndexType::Hnsw);
}

TEST(HnswCollection, EmptyIndexSearchReturnsNothing) {
  const auto c = make_hnsw(3);
  const auto r = c->search(std::vector<float>{1, 2, 3}, params(5));
  ASSERT_TRUE(r.ok());
  EXPECT_TRUE(r.value().empty());
  std::vector<Neighbor> out(5);
  EXPECT_EQ(c->search_into(std::vector<float>{1, 2, 3}, params(5), out).value(), 0U);
}

TEST(HnswCollection, SingleElement) {
  const auto c = make_hnsw(3);
  ASSERT_TRUE(c->add(42, std::vector<float>{1, 0, 0}).ok());
  const auto r = c->search(std::vector<float>{0, 1, 0}, params(10)).value();
  ASSERT_EQ(r.size(), 1U);
  EXPECT_EQ(r[0].id, 42U);
  EXPECT_FLOAT_EQ(r[0].distance, 2.0F);
}

TEST(HnswCollection, KLargerThanCountAndEfSmallerThanK) {
  constexpr std::uint32_t kDim = 4;
  const auto c = make_hnsw(kDim);
  vf::detail::Xoshiro256ss rng(5);
  const std::vector<float> rows = vf::test::random_matrix(rng, 30, kDim);
  for (ExternalId id = 0; id < 30; ++id) {
    ASSERT_TRUE(c->add(id, std::span<const float>(rows).subspan(id * kDim, kDim)).ok());
  }
  const std::vector<float> q = vf::test::random_vector(rng, kDim);
  const auto all = c->search(q, params(100)).value();
  ASSERT_EQ(all.size(), 30U) << "a 30-node graph is small enough to be explored fully";
  std::set<ExternalId> ids;
  for (std::size_t i = 0; i < all.size(); ++i) {
    ids.insert(all[i].id);
    if (i > 0) {
      EXPECT_LE(all[i - 1].distance, all[i].distance);
    }
  }
  EXPECT_EQ(ids.size(), 30U);
  // ef_search = 1 < k = 10: the beam is widened to k.
  EXPECT_EQ(c->search(q, params(10, 1)).value().size(), 10U);
}

TEST(HnswCollection, DimensionOne) {
  const auto c = make_hnsw(1);
  for (ExternalId id = 0; id < 200; ++id) {
    ASSERT_TRUE(c->add(id, std::vector<float>{static_cast<float>(id)}).ok());
  }
  const auto r = c->search(std::vector<float>{57.2F}, params(3, 32)).value();
  ASSERT_EQ(r.size(), 3U);
  EXPECT_EQ(r[0].id, 57U);
  EXPECT_EQ(r[1].id, 58U);
  EXPECT_EQ(r[2].id, 56U);
}

TEST(HnswCollection, DeleteEntryPointAndEverything) {
  constexpr std::uint32_t kDim = 8;
  vf::HnswParams hp;
  hp.M = 8;
  hp.ef_construction = 64;
  vf::test::HnswFixture f(kDim, Metric::L2, hp);
  vf::detail::Xoshiro256ss rng(9);
  f.add_rows(vf::test::random_matrix(rng, 300, kDim));
  const vf::InternalId entry = f.backend->graph().entry().id;
  f.remove(entry);
  const std::vector<float> at_entry(f.store.row(entry).begin(), f.store.row(entry).end());
  const std::vector<Neighbor> hits = f.search(at_entry, 10, 64);
  ASSERT_EQ(hits.size(), 10U);
  for (const Neighbor& n : hits) {
    EXPECT_NE(n.id, entry) << "a tombstoned entry point is still used for navigation only";
  }
  for (vf::InternalId id = 0; id < 300; ++id) {
    f.remove(id);
  }
  EXPECT_TRUE(f.search(at_entry, 10, 64).empty());
  expect_valid_graph(f);
}

TEST(HnswCollection, RemoveReinsertAndUpsert) {
  const auto c = make_hnsw(2);
  for (ExternalId id = 0; id < 50; ++id) {
    ASSERT_TRUE(c->add(id, std::vector<float>{static_cast<float>(id), 0.0F}).ok());
  }
  ASSERT_TRUE(c->remove(10).ok());
  EXPECT_FALSE(c->contains(10));
  EXPECT_EQ(c->remove(10).code(), ErrorCode::NotFound);
  auto r = c->search(std::vector<float>{10.0F, 0.0F}, params(1, 32)).value();
  ASSERT_EQ(r.size(), 1U);
  EXPECT_NE(r[0].id, 10U);

  ASSERT_TRUE(c->add(10, std::vector<float>{100.0F, 0.0F}).ok()) << "reinsert a deleted id";
  r = c->search(std::vector<float>{100.0F, 0.0F}, params(1, 32)).value();
  ASSERT_EQ(r.size(), 1U);
  EXPECT_EQ(r[0].id, 10U);

  EXPECT_EQ(c->add(20, std::vector<float>{-5.0F, 0.0F}).code(), ErrorCode::AlreadyExists);
  ASSERT_TRUE(c->add(20, std::vector<float>{-5.0F, 0.0F}, {.upsert = true}).ok());
  r = c->search(std::vector<float>{20.0F, 0.0F}, params(3, 32)).value();
  for (const Neighbor& n : r) {
    EXPECT_NE(n.id, 20U) << "the old vector of an upserted id must not be returned";
  }
  r = c->search(std::vector<float>{-5.0F, 0.0F}, params(1, 32)).value();
  ASSERT_EQ(r.size(), 1U);
  EXPECT_EQ(r[0].id, 20U);
  EXPECT_EQ(c->get(20).value(), (std::vector<float>{-5.0F, 0.0F}));

  const vf::CollectionStats st = c->stats();
  EXPECT_EQ(st.live_count, 50U);
  EXPECT_EQ(st.row_count, 52U);
  EXPECT_EQ(st.deleted_count, 2U);
  EXPECT_GT(st.memory.index_bytes, 0U);
}

TEST(HnswCollection, SearchBatchMatchesSingleSearches) {
  constexpr std::uint32_t kDim = 6;
  const auto c = make_hnsw(kDim, Metric::Cosine);
  vf::detail::Xoshiro256ss rng(11);
  const std::vector<float> rows = vf::test::random_matrix(rng, 500, kDim);
  std::vector<ExternalId> ids(500);
  for (std::size_t i = 0; i < ids.size(); ++i) {
    ids[i] = 1000 + i;
  }
  ASSERT_EQ(c->add_batch(ids, rows).value(), 500U);
  constexpr std::size_t kQueries = 7;
  constexpr std::uint32_t kK = 5;
  const std::vector<float> queries = vf::test::random_matrix(rng, kQueries, kDim);
  std::vector<ExternalId> out_ids(kQueries * kK);
  std::vector<float> out_dist(kQueries * kK);
  std::vector<std::uint32_t> counts(kQueries);
  ASSERT_TRUE(c->search_batch(queries, kQueries, params(kK, 40), out_ids, out_dist, counts).ok());
  for (std::size_t q = 0; q < kQueries; ++q) {
    const auto single =
        c->search(std::span<const float>(queries).subspan(q * kDim, kDim), params(kK, 40)).value();
    ASSERT_EQ(counts[q], single.size());
    for (std::size_t j = 0; j < single.size(); ++j) {
      EXPECT_EQ(out_ids[(q * kK) + j], single[j].id);
      EXPECT_EQ(out_dist[(q * kK) + j], single[j].distance);
    }
  }
}

// 1 000 identical vectors (§9.7): every distance is 0, so neighbour selection has no geometry to
// work with. Searches must still return k results and the graph must stay fully reachable.
TEST(HnswDuplicates, IdenticalVectors) {
  vf::test::HnswFixture f(4, Metric::L2, vf::HnswParams{});
  const std::vector<float> v = {0.25F, -1.0F, 3.0F, 0.5F};
  for (int i = 0; i < 1000; ++i) {
    f.add(v);
  }
  const std::vector<Neighbor> hits = f.search(v, 10);
  ASSERT_EQ(hits.size(), 10U);
  for (const Neighbor& n : hits) {
    EXPECT_EQ(n.distance, 0.0F);
  }
  expect_valid_graph(f);
}

TEST(HnswDuplicates, NearIdenticalVectors) {
  constexpr std::uint32_t kDim = 4;
  vf::test::HnswFixture f(kDim, Metric::L2, vf::HnswParams{});
  vf::detail::Xoshiro256ss rng(3);
  const std::vector<float> centre = {0.25F, -1.0F, 3.0F, 0.5F};
  std::vector<std::vector<float>> rows;
  for (int i = 0; i < 1000; ++i) {
    std::vector<float> v = centre;
    for (float& x : v) {
      x += vf::detail::uniform_float(rng(), -1e-4F, 1e-4F);
    }
    rows.push_back(v);
    f.add(v);
  }
  const std::vector<Neighbor> hits = f.search(centre, 10);
  ASSERT_EQ(hits.size(), 10U);
  expect_valid_graph(f);

  // Recall against brute force on this degenerate cluster.
  std::vector<vf::test::RefItem> items;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    items.push_back({i, i, rows[i]});
  }
  const auto ranking = vf::test::reference_ranking(Metric::L2, false, centre, items);
  std::set<ExternalId> exact;
  for (std::size_t i = 0; i < 10; ++i) {
    exact.insert(ranking[i].id);
  }
  std::size_t found = 0;
  for (const Neighbor& n : f.search(centre, 10, 200)) {
    found += exact.contains(n.id) ? 1U : 0U;
  }
  EXPECT_GE(found, 9U);
}

TEST(HnswBackend, SmallChunksAndDeepLevels) {
  // M = 2 produces many upper levels; tiny chunks exercise every storage boundary.
  vf::HnswParams hp;
  hp.M = 2;
  hp.ef_construction = 16;
  hp.max_level = 8;
  vf::test::HnswFixture f(3, Metric::L2, hp, {.nodes_per_chunk = 8, .arena_chunk_words = 32},
                          /*rows_per_chunk=*/16);
  vf::detail::Xoshiro256ss rng(21);
  f.add_rows(vf::test::random_matrix(rng, 2000, 3));
  const vf::Status st = f.validator().check_invariants();
  ASSERT_TRUE(st.ok()) << st.to_string();
  EXPECT_GE(f.backend->graph().entry().level, 4U);
  const std::vector<std::uint64_t> hist = f.validator().level_histogram();
  EXPECT_GT(hist[1], 0U);
  const std::vector<float> q = vf::test::random_vector(rng, 3);
  EXPECT_EQ(f.search(q, 10, 64).size(), 10U);
}

}  // namespace
