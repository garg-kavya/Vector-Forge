// Concurrent const access (docs/DESIGN.md §11; Collection thread-safety contract): const member
// functions may run concurrently with each other. Several threads issue search, search_into,
// search_batch, get, contains and stats at the same time; every result must equal the result the
// same call produced single-threaded (queries are deterministic). Run under TSan in CI/local
// sanitizer builds to detect data races.

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include <vectorforge/collection.hpp>

#include "support/test_data.hpp"

namespace {

using vf::Collection;
using vf::ExternalId;
using vf::IndexType;
using vf::Neighbor;

constexpr std::uint32_t kDim = 16;
constexpr std::size_t kRows = 2000;
constexpr std::size_t kQueries = 64;
constexpr std::uint32_t kK = 10;
constexpr std::size_t kThreads = 8;

class ConcurrentReads : public testing::TestWithParam<IndexType> {};

TEST_P(ConcurrentReads, ResultsMatchSingleThreaded) {
  vf::CollectionConfig cfg;
  cfg.dim = kDim;
  cfg.metric = vf::Metric::Cosine;
  cfg.index = GetParam();
  cfg.hnsw.M = 8;
  cfg.hnsw.ef_construction = 64;
  const std::unique_ptr<Collection> c = Collection::create(cfg).value();
  const vf::test::ClusteredData data(9, kDim, 20, 0.2F);
  const std::vector<float> rows = data.rows(1, kRows);
  std::vector<ExternalId> ids(kRows);
  for (std::size_t i = 0; i < kRows; ++i) {
    ids[i] = 10 * i;
  }
  ASSERT_EQ(c->add_batch(ids, rows).value(), kRows);
  for (std::size_t i = 0; i < kRows; i += 7) {
    ASSERT_TRUE(c->remove(ids[i]).ok());
  }
  const std::vector<float> queries = data.rows(2, kQueries);
  vf::SearchParams params;
  params.k = kK;
  params.ef_search = 48;

  auto query = [&](std::size_t q) {
    return std::span<const float>(queries).subspan(q * kDim, kDim);
  };
  std::vector<std::vector<Neighbor>> expected(kQueries);
  for (std::size_t q = 0; q < kQueries; ++q) {
    expected[q] = c->search(query(q), params).value();
  }
  const std::size_t expected_size = c->size();

  std::atomic<std::size_t> mismatches{0};
  std::atomic<std::size_t> calls{0};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (std::size_t t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      std::vector<Neighbor> out(kK);
      std::vector<ExternalId> batch_ids(4 * kK);
      std::vector<float> batch_dist(4 * kK);
      std::vector<std::uint32_t> counts(4);
      for (std::size_t round = 0; round < 40; ++round) {
        const std::size_t q = (t * 13 + round) % kQueries;
        switch ((t + round) % 5) {
          case 0: {
            const auto r = c->search(query(q), params);
            mismatches += (!r.ok() || r.value() != expected[q]) ? 1U : 0U;
            break;
          }
          case 1: {
            const auto n = c->search_into(query(q), params, out);
            const bool same =
                n.ok() && n.value() == expected[q].size() &&
                std::equal(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(n.value()),
                           expected[q].begin());
            mismatches += same ? 0U : 1U;
            break;
          }
          case 2: {
            const std::size_t first = q % (kQueries - 4);
            const vf::Status st =
                c->search_batch(std::span<const float>(queries).subspan(first * kDim, 4 * kDim), 4,
                                params, batch_ids, batch_dist, counts);
            bool same = st.ok();
            for (std::size_t b = 0; same && b < 4; ++b) {
              const auto& e = expected[first + b];
              same = counts[b] == e.size();
              for (std::size_t j = 0; same && j < e.size(); ++j) {
                same =
                    batch_ids[(b * kK) + j] == e[j].id && batch_dist[(b * kK) + j] == e[j].distance;
              }
            }
            mismatches += same ? 0U : 1U;
            break;
          }
          case 3: {
            const ExternalId id = ids[(q * 31) % kRows];
            const bool live = ((q * 31) % kRows) % 7 != 0;
            mismatches += (c->contains(id) != live || c->get(id).ok() != live) ? 1U : 0U;
            break;
          }
          default: {
            const vf::CollectionStats st = c->stats();
            mismatches += (st.live_count != expected_size || st.row_count != kRows) ? 1U : 0U;
            break;
          }
        }
        ++calls;
      }
    });
  }
  for (std::thread& th : threads) {
    th.join();
  }
  EXPECT_EQ(calls.load(), kThreads * 40);
  EXPECT_EQ(mismatches.load(), 0U);
}

INSTANTIATE_TEST_SUITE_P(Indexes, ConcurrentReads,
                         testing::Values(IndexType::Flat, IndexType::Hnsw),
                         [](const testing::TestParamInfo<IndexType>& param_info) {
                           return std::string(vf::to_string(param_info.param));
                         });

}  // namespace
