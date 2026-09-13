#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include <vectorforge/types.hpp>

#include "core/rng.hpp"
#include "core/vector_ops.hpp"
#include "index/flat_backend.hpp"
#include "simd/kernels.hpp"
#include "storage/tombstones.hpp"
#include "storage/vector_store.hpp"
#include "support/brute_force_reference.hpp"
#include "support/test_data.hpp"

namespace {

using vf::InternalId;
using vf::Metric;
using vf::Neighbor;
using vf::detail::FlatBackend;
using vf::detail::QueryView;
using vf::detail::SearchKnobs;
using vf::detail::TombstoneSet;
using vf::detail::VectorStore;

struct Fixture {
  VectorStore store;
  TombstoneSet deleted;
  std::vector<std::vector<float>> raw;  // as inserted (before normalisation)

  Fixture(std::uint32_t dim, std::size_t rows_per_chunk)
      : store(VectorStore::create({.dim = dim, .rows_per_chunk = rows_per_chunk}).value()) {}

  void add(std::vector<float> v, bool normalize) {
    raw.push_back(v);
    if (normalize) {
      ASSERT_TRUE(vf::detail::normalize_inplace(v, vf::detail::kernels()).ok());
    }
    ASSERT_TRUE(store.append(v).ok());
    deleted.ensure_size(store.size());
  }

  [[nodiscard]] std::vector<vf::test::RefItem> live_items() const {
    std::vector<vf::test::RefItem> items;
    for (std::size_t i = 0; i < raw.size(); ++i) {
      if (!deleted.test(static_cast<InternalId>(i))) {
        items.push_back({static_cast<vf::ExternalId>(i), i, raw[i]});
      }
    }
    return items;
  }
};

std::vector<Neighbor> run(const FlatBackend& backend, std::span<const float> query, std::uint32_t k,
                          bool normalized) {
  QueryView view{.data = query.data(), .inv_norm = 1.0F};
  if (normalized) {
    view.inv_norm = vf::detail::inverse_norm(query, vf::detail::kernels()).value();
  }
  std::vector<Neighbor> out(k);
  const std::size_t count = backend.search(view, SearchKnobs{.k = k}, out);
  out.resize(count);
  return out;
}

TEST(FlatBackend, ExactOnIntegerDataAcrossChunkAndBlockBoundaries) {
  constexpr std::uint32_t kDim = 12;
  // rows_per_chunk 256 exercises chunk boundaries; 4096 exercises 1024-row block boundaries.
  for (const std::size_t rows_per_chunk : {256U, 4096U}) {
    for (const Metric metric : {Metric::L2, Metric::InnerProduct}) {
      Fixture f(kDim, rows_per_chunk);
      vf::detail::Xoshiro256ss rng(rows_per_chunk + static_cast<std::uint64_t>(metric));
      for (int i = 0; i < 2600; ++i) {
        f.add(vf::test::integer_vector(rng, kDim, -3, 3), false);  // narrow range: many exact ties
      }
      const FlatBackend backend(f.store, f.deleted, metric, false, vf::detail::kernels());
      EXPECT_EQ(backend.type(), vf::IndexType::Flat);
      for (int q = 0; q < 12; ++q) {
        const std::vector<float> query = vf::test::integer_vector(rng, kDim, -3, 3);
        const auto ranking = vf::test::reference_ranking(metric, false, query, f.live_items());
        for (const std::uint32_t k : {1U, 10U, 32U, 33U, 500U, 2600U, 3000U}) {
          SCOPED_TRACE(testing::Message() << "chunk=" << rows_per_chunk << " metric="
                                          << vf::to_string(metric) << " q=" << q << " k=" << k);
          vf::test::expect_exact_topk(run(backend, query, k, false), ranking, k);
        }
      }
    }
  }
}

TEST(FlatBackend, SkipsTombstones) {
  constexpr std::uint32_t kDim = 4;
  Fixture f(kDim, 16);
  vf::detail::Xoshiro256ss rng(9);
  for (int i = 0; i < 200; ++i) {
    f.add(vf::test::integer_vector(rng, kDim), false);
  }
  for (InternalId id = 0; id < 200; id += 3) {
    f.deleted.set(id);
  }
  const FlatBackend backend(f.store, f.deleted, Metric::L2, false, vf::detail::kernels());
  const std::vector<float> query = vf::test::integer_vector(rng, kDim);
  const auto ranking = vf::test::reference_ranking(Metric::L2, false, query, f.live_items());
  for (const std::uint32_t k : {1U, 20U, 64U, 200U}) {
    const auto result = run(backend, query, k, false);
    vf::test::expect_exact_topk(result, ranking, k);
    for (const Neighbor& n : result) {
      EXPECT_NE(n.id % 3, 0U) << "tombstoned row returned";
    }
  }
  for (InternalId id = 0; id < 200; ++id) {
    f.deleted.set(id);
  }
  EXPECT_TRUE(run(backend, query, 10, false).empty());
}

TEST(FlatBackend, EmptyStoreReturnsNothing) {
  Fixture f(3, 4);
  const FlatBackend backend(f.store, f.deleted, Metric::L2, false, vf::detail::kernels());
  const std::vector<float> query = {1.0F, 2.0F, 3.0F};
  EXPECT_TRUE(run(backend, query, 5, false).empty());
}

TEST(FlatBackend, NormalizedMetricsMatchReferenceWithinTolerance) {
  constexpr std::uint32_t kDim = 24;
  for (const Metric metric : {Metric::Cosine, Metric::L2, Metric::InnerProduct}) {
    Fixture f(kDim, 128);
    vf::detail::Xoshiro256ss rng(100 + static_cast<std::uint64_t>(metric));
    for (int i = 0; i < 1500; ++i) {
      f.add(vf::test::random_vector(rng, kDim), true);
    }
    const FlatBackend backend(f.store, f.deleted, metric, true, vf::detail::kernels());
    for (int q = 0; q < 10; ++q) {
      const std::vector<float> query = vf::test::random_vector(rng, kDim, -3.0F, 3.0F);
      const auto ranking = vf::test::reference_ranking(metric, true, query, f.live_items());
      for (const std::uint32_t k : {1U, 16U, 100U}) {
        SCOPED_TRACE(testing::Message() << "metric=" << vf::to_string(metric) << " k=" << k);
        vf::test::expect_tolerant_topk(run(backend, query, k, true), ranking, k, 1e-5);
      }
    }
  }
}

TEST(FlatBackend, NormalizedDistancesAreClampedNonNegative) {
  Fixture f(3, 4);
  f.add({1.0F, 2.0F, 3.0F}, true);
  for (const Metric metric : {Metric::Cosine, Metric::L2}) {
    const FlatBackend backend(f.store, f.deleted, metric, true, vf::detail::kernels());
    const std::vector<float> query = {2.0F, 4.0F, 6.0F};  // same direction
    const auto result = run(backend, query, 1, true);
    ASSERT_EQ(result.size(), 1U);
    EXPECT_GE(result[0].distance, 0.0F);
    EXPECT_LT(result[0].distance, 1e-6F);
  }
}

}  // namespace
