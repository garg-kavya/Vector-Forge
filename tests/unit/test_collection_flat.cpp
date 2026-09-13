#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include <vectorforge/collection.hpp>

#include "core/rng.hpp"
#include "support/brute_force_reference.hpp"
#include "support/test_data.hpp"

namespace {

using vf::Collection;
using vf::CollectionConfig;
using vf::ErrorCode;
using vf::IndexType;
using vf::Metric;
using vf::Neighbor;
using vf::SearchParams;

std::unique_ptr<Collection> make(std::uint32_t dim, Metric metric, bool normalize = false) {
  CollectionConfig cfg;
  cfg.dim = dim;
  cfg.metric = metric;
  cfg.normalize = normalize;
  cfg.index = IndexType::Flat;
  auto c = Collection::create(cfg);
  EXPECT_TRUE(c.ok()) << c.status().to_string();
  return std::move(c).value();
}

SearchParams k_of(std::uint32_t k) {
  SearchParams p;
  p.k = k;
  return p;
}

TEST(CollectionFlat, CreateValidation) {
  CollectionConfig cfg;
  EXPECT_EQ(Collection::create(cfg).status().code(), ErrorCode::InvalidArgument);  // dim 0
  cfg.dim = 4;
  cfg.index = static_cast<IndexType>(7);
  EXPECT_EQ(Collection::create(cfg).status().code(), ErrorCode::InvalidArgument);
  cfg.index = IndexType::Flat;
  cfg.hnsw.M = 1;  // HNSW parameters are not validated for Flat collections

  const auto ok = Collection::create(cfg);
  ASSERT_TRUE(ok.ok());
  EXPECT_EQ(ok.value()->config().dim, 4U);
  EXPECT_EQ(ok.value()->size(), 0U);
}

TEST(CollectionFlat, AddGetContainsRemove) {
  auto c = make(3, Metric::L2);
  EXPECT_TRUE(c->add(10, std::vector<float>{1, 2, 3}).ok());
  EXPECT_TRUE(c->add(20, std::vector<float>{4, 5, 6}).ok());
  EXPECT_EQ(c->size(), 2U);
  EXPECT_TRUE(c->contains(10));
  EXPECT_FALSE(c->contains(30));
  EXPECT_EQ(c->get(20).value(), (std::vector<float>{4, 5, 6}));
  EXPECT_EQ(c->get(30).status().code(), ErrorCode::NotFound);

  EXPECT_EQ(c->add(10, std::vector<float>{0, 0, 0}).code(), ErrorCode::AlreadyExists);
  EXPECT_EQ(c->get(10).value(), (std::vector<float>{1, 2, 3})) << "failed add changes nothing";

  EXPECT_TRUE(c->remove(10).ok());
  EXPECT_FALSE(c->contains(10));
  EXPECT_EQ(c->remove(10).code(), ErrorCode::NotFound);
  EXPECT_EQ(c->size(), 1U);
  EXPECT_TRUE(c->add(10, std::vector<float>{7, 8, 9}).ok()) << "removed id can be reinserted";
  EXPECT_EQ(c->get(10).value(), (std::vector<float>{7, 8, 9}));

  const vf::CollectionStats st = c->stats();
  EXPECT_EQ(st.live_count, 2U);
  EXPECT_EQ(st.deleted_count, 1U);
  EXPECT_EQ(st.row_count, 3U);
  EXPECT_EQ(st.dim, 3U);
  EXPECT_EQ(st.index, IndexType::Flat);
  EXPECT_EQ(st.simd, vf::active_simd_level());
  EXPECT_GE(st.memory.vectors_bytes, 3 * 3 * sizeof(float));
  EXPECT_EQ(st.memory.index_bytes, 0U);
  EXPECT_GT(st.memory.total_bytes(), st.memory.vectors_bytes);
}

TEST(CollectionFlat, AddValidation) {
  auto c = make(3, Metric::L2);
  EXPECT_EQ(c->add(1, std::vector<float>{1, 2}).code(), ErrorCode::DimensionMismatch);
  EXPECT_EQ(c->add(1, std::vector<float>{1, std::numeric_limits<float>::quiet_NaN(), 3}).code(),
            ErrorCode::InvalidArgument);
  EXPECT_EQ(c->add(vf::kInvalidExternalId, std::vector<float>{1, 2, 3}).code(),
            ErrorCode::InvalidArgument);
  EXPECT_EQ(c->size(), 0U);
  EXPECT_EQ(c->stats().row_count, 0U) << "rejected inserts store nothing";

  auto cos = make(3, Metric::Cosine);
  EXPECT_EQ(cos->add(1, std::vector<float>{0, 0, 0}).code(), ErrorCode::InvalidArgument);
  EXPECT_EQ(cos->stats().row_count, 0U);
}

TEST(CollectionFlat, UpsertReplacesVector) {
  auto c = make(2, Metric::L2);
  ASSERT_TRUE(c->add(1, std::vector<float>{0, 0}).ok());
  ASSERT_TRUE(c->add(2, std::vector<float>{5, 5}).ok());
  ASSERT_TRUE(c->add(1, std::vector<float>{10, 10}, {.upsert = true}).ok());
  EXPECT_EQ(c->get(1).value(), (std::vector<float>{10, 10}));
  EXPECT_EQ(c->size(), 2U);
  EXPECT_EQ(c->stats().deleted_count, 1U);
  const auto hits = c->search(std::vector<float>{0, 0}, k_of(5)).value();
  ASSERT_EQ(hits.size(), 2U);
  EXPECT_EQ(hits[0], (Neighbor{2, 50.0F}));
  EXPECT_EQ(hits[1], (Neighbor{1, 200.0F})) << "old vector of id 1 is not searchable";
  // Upsert of a new id behaves like insert.
  EXPECT_TRUE(c->add(3, std::vector<float>{1, 1}, {.upsert = true}).ok());
  EXPECT_EQ(c->size(), 3U);
}

TEST(CollectionFlat, NormalizedCollectionsStoreUnitVectors) {
  auto c = make(2, Metric::Cosine);
  ASSERT_TRUE(c->add(1, std::vector<float>{3, 4}).ok());
  const auto v = c->get(1).value();
  EXPECT_FLOAT_EQ(v[0], 0.6F);
  EXPECT_FLOAT_EQ(v[1], 0.8F);
  EXPECT_TRUE(c->stats().normalized);

  auto l2n = make(2, Metric::L2, /*normalize=*/true);
  ASSERT_TRUE(l2n->add(1, std::vector<float>{0, 9}).ok());
  ASSERT_TRUE(l2n->add(2, std::vector<float>{9, 0.5F}).ok());
  const auto hits = l2n->search(std::vector<float>{0, 0.01F}, k_of(2)).value();
  ASSERT_EQ(hits.size(), 2U);
  EXPECT_EQ(hits[0].id, 1U);
  EXPECT_NEAR(hits[0].distance, 0.0F, 1e-6F);
}

TEST(CollectionFlat, SearchSemantics) {
  auto c = make(1, Metric::L2);
  for (int i = 0; i < 10; ++i) {
    ASSERT_TRUE(
        c->add(static_cast<vf::ExternalId>(100 + i), std::vector<float>{static_cast<float>(i)})
            .ok());
  }
  // Ties at equal distance resolve by insertion order: 104 (d=1) before 106 (d=1).
  const auto hits = c->search(std::vector<float>{5.0F}, k_of(3)).value();
  ASSERT_EQ(hits.size(), 3U);
  EXPECT_EQ(hits[0], (Neighbor{105, 0.0F}));
  EXPECT_EQ(hits[1], (Neighbor{104, 1.0F}));
  EXPECT_EQ(hits[2], (Neighbor{106, 1.0F}));

  EXPECT_EQ(c->search(std::vector<float>{0.0F}, k_of(1000)).value().size(), 10U) << "k > size";
  EXPECT_EQ(c->search(std::vector<float>{0.0F}, k_of(0)).status().code(),
            ErrorCode::InvalidArgument);
  EXPECT_EQ(c->search(std::vector<float>{0.0F}, k_of(SearchParams::kMaxK + 1)).status().code(),
            ErrorCode::InvalidArgument);
  EXPECT_EQ(c->search(std::vector<float>{0.0F, 1.0F}, k_of(1)).status().code(),
            ErrorCode::DimensionMismatch);
  EXPECT_EQ(c->search(std::vector<float>{std::numeric_limits<float>::infinity()}, k_of(1))
                .status()
                .code(),
            ErrorCode::InvalidArgument);
  SearchParams bad_ef = k_of(1);
  bad_ef.ef_search = 0;
  EXPECT_EQ(c->search(std::vector<float>{0.0F}, bad_ef).status().code(),
            ErrorCode::InvalidArgument);
  SearchParams ef = k_of(1);
  ef.ef_search = 64;  // accepted and ignored by Flat
  EXPECT_TRUE(c->search(std::vector<float>{0.0F}, ef).ok());

  auto empty = make(2, Metric::InnerProduct);
  EXPECT_TRUE(empty->search(std::vector<float>{1, 1}, k_of(5)).value().empty());

  auto cos = make(2, Metric::Cosine);
  ASSERT_TRUE(cos->add(1, std::vector<float>{1, 0}).ok());
  EXPECT_EQ(cos->search(std::vector<float>{0, 0}, k_of(1)).status().code(),
            ErrorCode::InvalidArgument)
      << "zero query in a normalised collection";
}

TEST(CollectionFlat, SearchInto) {
  auto c = make(2, Metric::L2);
  ASSERT_TRUE(c->add(1, std::vector<float>{1, 1}).ok());
  ASSERT_TRUE(c->add(2, std::vector<float>{2, 2}).ok());
  std::vector<Neighbor> out(4, Neighbor{999, -1.0F});
  const auto n = c->search_into(std::vector<float>{0, 0}, k_of(4), out);
  ASSERT_TRUE(n.ok());
  ASSERT_EQ(n.value(), 2U);
  EXPECT_EQ(out[0], (Neighbor{1, 2.0F}));
  EXPECT_EQ(out[1], (Neighbor{2, 8.0F}));
  std::vector<Neighbor> small(1);
  EXPECT_EQ(c->search_into(std::vector<float>{0, 0}, k_of(2), small).status().code(),
            ErrorCode::InvalidArgument);
}

TEST(CollectionFlat, SearchBatchMatchesSingleQueriesAndPads) {
  constexpr std::uint32_t kDim = 8;
  auto c = make(kDim, Metric::InnerProduct);
  vf::detail::Xoshiro256ss rng(21);
  for (vf::ExternalId id = 0; id < 5; ++id) {
    ASSERT_TRUE(c->add(id, vf::test::integer_vector(rng, kDim)).ok());
  }
  constexpr std::size_t kQueries = 4;
  constexpr std::uint32_t kK = 7;  // more than stored vectors -> padding
  const std::vector<float> queries = vf::test::integer_matrix(rng, kQueries, kDim);
  std::vector<vf::ExternalId> ids(kQueries * kK);
  std::vector<float> dists(kQueries * kK);
  std::vector<std::uint32_t> counts(kQueries);
  ASSERT_TRUE(c->search_batch(queries, kQueries, k_of(kK), ids, dists, counts).ok());
  for (std::size_t q = 0; q < kQueries; ++q) {
    const auto single =
        c->search(std::span<const float>(queries).subspan(q * kDim, kDim), k_of(kK)).value();
    ASSERT_EQ(counts[q], single.size());
    for (std::size_t j = 0; j < kK; ++j) {
      if (j < single.size()) {
        EXPECT_EQ(ids[(q * kK) + j], single[j].id);
        EXPECT_EQ(dists[(q * kK) + j], single[j].distance);
      } else {
        EXPECT_EQ(ids[(q * kK) + j], vf::kInvalidExternalId);
        EXPECT_TRUE(std::isinf(dists[(q * kK) + j]));
      }
    }
  }

  // Errors: short outputs, wrong query size, invalid query anywhere in the batch.
  std::vector<vf::ExternalId> short_ids(3);
  EXPECT_EQ(c->search_batch(queries, kQueries, k_of(kK), short_ids, dists, counts).code(),
            ErrorCode::InvalidArgument);
  EXPECT_EQ(c->search_batch(queries, kQueries + 1, k_of(kK), ids, dists, counts).code(),
            ErrorCode::DimensionMismatch);
  std::vector<float> bad = queries;
  bad.back() = std::numeric_limits<float>::quiet_NaN();
  EXPECT_EQ(c->search_batch(bad, kQueries, k_of(kK), ids, dists, counts).code(),
            ErrorCode::InvalidArgument);
  EXPECT_TRUE(c->search_batch(std::vector<float>{}, 0, k_of(kK), ids, dists, counts).ok());
}

TEST(CollectionFlat, AddBatch) {
  auto c = make(2, Metric::L2);
  const std::vector<vf::ExternalId> ids = {5, 6, 7};
  const std::vector<float> rows = {0, 0, 1, 1, 2, 2};
  const auto added = c->add_batch(ids, rows);
  ASSERT_TRUE(added.ok());
  EXPECT_EQ(added.value(), 3U);
  EXPECT_EQ(c->get(7).value(), (std::vector<float>{2, 2}));
  EXPECT_EQ(c->add_batch({}, {}).value(), 0U);

  // All-or-nothing validation.
  EXPECT_EQ(c->add_batch(std::vector<vf::ExternalId>{8, 8}, std::vector<float>{0, 0, 1, 1})
                .status()
                .code(),
            ErrorCode::InvalidArgument);
  EXPECT_EQ(c->add_batch(std::vector<vf::ExternalId>{8, 5}, std::vector<float>{0, 0, 1, 1})
                .status()
                .code(),
            ErrorCode::AlreadyExists);
  EXPECT_EQ(
      c->add_batch(std::vector<vf::ExternalId>{8, 9}, std::vector<float>{0, 0, 1}).status().code(),
      ErrorCode::DimensionMismatch);
  EXPECT_EQ(c->add_batch(std::vector<vf::ExternalId>{8, 9},
                         std::vector<float>{0, 0, 1, std::numeric_limits<float>::infinity()})
                .status()
                .code(),
            ErrorCode::InvalidArgument);
  EXPECT_EQ(c->add_batch(std::vector<vf::ExternalId>{8, vf::kInvalidExternalId},
                         std::vector<float>{0, 0, 1, 1})
                .status()
                .code(),
            ErrorCode::InvalidArgument);
  EXPECT_FALSE(c->contains(8)) << "rejected batch inserted nothing";
  EXPECT_EQ(c->stats().row_count, 3U);

  // Upsert batch.
  ASSERT_TRUE(c->add_batch(std::vector<vf::ExternalId>{5, 8}, std::vector<float>{9, 9, 8, 8},
                           {.upsert = true})
                  .ok());
  EXPECT_EQ(c->get(5).value(), (std::vector<float>{9, 9}));
  EXPECT_EQ(c->size(), 4U);

  auto cos = make(2, Metric::Cosine);
  EXPECT_EQ(cos->add_batch(std::vector<vf::ExternalId>{1, 2}, std::vector<float>{1, 0, 0, 0})
                .status()
                .code(),
            ErrorCode::InvalidArgument);
  EXPECT_EQ(cos->size(), 0U) << "zero row rejected before any insert";
}

TEST(CollectionFlat, ExactAgainstReferenceWithManyTies) {
  constexpr std::uint32_t kDim = 6;
  for (const Metric metric : {Metric::L2, Metric::InnerProduct}) {
    auto c = make(kDim, metric);
    vf::detail::Xoshiro256ss rng(55);
    std::vector<std::vector<float>> vecs;
    std::vector<vf::test::RefItem> items;
    for (vf::ExternalId id = 0; id < 3000; ++id) {
      vecs.push_back(vf::test::integer_vector(rng, kDim, -2, 2));
      ASSERT_TRUE(c->add(id * 13, vecs.back()).ok());
    }
    for (std::size_t i = 0; i < vecs.size(); ++i) {
      items.push_back({static_cast<vf::ExternalId>(i) * 13, i, vecs[i]});
    }
    for (int q = 0; q < 20; ++q) {
      const std::vector<float> query = vf::test::integer_vector(rng, kDim, -2, 2);
      const auto ranking = vf::test::reference_ranking(metric, false, query, items);
      for (const std::uint32_t k : {1U, 7U, 40U, 3000U}) {
        vf::test::expect_exact_topk(c->search(query, k_of(k)).value(), ranking, k);
      }
    }
  }
}

}  // namespace
