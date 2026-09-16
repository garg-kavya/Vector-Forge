// Level A concurrency (docs/DESIGN.md §11.3, §11.7; docs/concurrency.md): reader threads search
// while writer threads insert and remove. Invariants checked on every result:
//   - results are sorted by ascending distance and contain no duplicate id;
//   - every id is one that was inserted at some point;
//   - no id whose remove() had returned before the query started is returned.
// A second test adds compaction and parallel batch operations to the mix. Run under TSan.

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <vectorforge/collection.hpp>
#include <vectorforge/thread_pool.hpp>

#include "support/test_data.hpp"

namespace {

using vf::Collection;
using vf::ExternalId;
using vf::IndexType;
using vf::Neighbor;

constexpr std::uint32_t kDim = 8;
constexpr std::size_t kPreloaded = 1500;
constexpr ExternalId kWriterBase = 1000000;
constexpr std::uint32_t kK = 10;

struct Fixture {
  std::unique_ptr<Collection> collection;
  vf::test::ClusteredData data{17, kDim, 12, 0.3F};
  // removed[i] becomes true once remove(i) has returned (preloaded ids only).
  std::vector<std::atomic<bool>> removed = std::vector<std::atomic<bool>>(kPreloaded);

  explicit Fixture(IndexType index) {
    vf::CollectionConfig cfg;
    cfg.dim = kDim;
    cfg.metric = vf::Metric::L2;
    cfg.index = index;
    cfg.hnsw.M = 8;
    cfg.hnsw.ef_construction = 48;
    collection = Collection::create(cfg).value();
    std::vector<ExternalId> ids(kPreloaded);
    for (std::size_t i = 0; i < kPreloaded; ++i) {
      ids[i] = i;
    }
    EXPECT_EQ(collection->add_batch(ids, data.rows(1, kPreloaded)).value(), kPreloaded);
  }

  // True if `id` could legitimately appear in a result.
  [[nodiscard]] static bool known(ExternalId id) {
    return id < kPreloaded || (id >= kWriterBase && id < kWriterBase + 10000000);
  }
};

// Checks one result against the invariants; `removed_before` lists ids removed before the query.
bool valid(std::span<const Neighbor> hits, const std::unordered_set<ExternalId>& removed_before) {
  std::unordered_set<ExternalId> seen;
  for (std::size_t i = 0; i < hits.size(); ++i) {
    if (i > 0 && hits[i].distance < hits[i - 1].distance) {
      return false;
    }
    if (!Fixture::known(hits[i].id) || removed_before.contains(hits[i].id) ||
        !seen.insert(hits[i].id).second) {
      return false;
    }
  }
  return true;
}

std::unordered_set<ExternalId> snapshot_removed(const Fixture& f) {
  std::unordered_set<ExternalId> out;
  for (std::size_t i = 0; i < kPreloaded; ++i) {
    if (f.removed[i].load()) {
      out.insert(i);
    }
  }
  return out;
}

class ConcurrentSearch : public testing::TestWithParam<IndexType> {};

TEST_P(ConcurrentSearch, ReadersSeeConsistentResultsWhileWritersMutate) {
  Fixture f(GetParam());
  vf::ThreadPool pool(3);
  std::atomic<bool> writers_done{false};
  std::atomic<std::size_t> violations{0};
  std::atomic<std::size_t> queries{0};
  std::atomic<std::size_t> inserted{0};
  std::atomic<std::size_t> removals{0};

  std::vector<std::thread> writers;
  for (std::size_t w = 0; w < 2; ++w) {
    writers.emplace_back([&, w] {
      const std::vector<float> rows = f.data.rows(100 + w, 300);
      for (std::size_t i = 0; i < 300; ++i) {
        const ExternalId id = kWriterBase + (w * 1000000) + i;
        if (i % 50 == 0) {
          // A batch of five, inserted with the pool.
          std::vector<ExternalId> ids;
          std::vector<float> batch;
          for (std::size_t b = 0; b < 5 && i + b < 300; ++b) {
            ids.push_back(id + 5000 + b);
            const auto row = std::span<const float>(rows).subspan((i + b) * kDim, kDim);
            batch.insert(batch.end(), row.begin(), row.end());
          }
          const auto added = f.collection->add_batch(ids, batch, {}, &pool);
          violations += added.ok() && added.value() == ids.size() ? 0U : 1U;
          inserted += ids.size();
        }
        violations +=
            f.collection->add(id, std::span<const float>(rows).subspan(i * kDim, kDim)).ok() ? 0U
                                                                                             : 1U;
        ++inserted;
        // Writer w removes preloaded ids congruent to w modulo 2.
        const std::size_t victim = (i * 2) + w;
        if (victim < kPreloaded && victim % 3 == 0) {
          violations += f.collection->remove(victim).ok() ? 0U : 1U;
          f.removed[victim].store(true);
          ++removals;
        }
      }
    });
  }

  std::vector<std::thread> readers;
  const std::vector<float> query_rows = f.data.rows(7, 64);
  for (std::size_t r = 0; r < 4; ++r) {
    readers.emplace_back([&, r] {
      vf::SearchParams params;
      params.k = kK;
      params.ef_search = 32;
      std::vector<Neighbor> out(kK);
      std::vector<ExternalId> batch_ids(4 * kK);
      std::vector<float> batch_dist(4 * kK);
      std::vector<std::uint32_t> counts(4);
      for (std::size_t round = 0; !writers_done.load() || round < 50; ++round) {
        const std::size_t q = (r * 11 + round) % 60;
        const std::unordered_set<ExternalId> removed_before = snapshot_removed(f);
        const auto query = std::span<const float>(query_rows).subspan(q * kDim, kDim);
        if (round % 3 == 2) {
          const vf::Status st = f.collection->search_batch(
              std::span<const float>(query_rows).subspan(q * kDim, 4 * kDim), 4, params, batch_ids,
              batch_dist, counts, &pool);
          bool ok = st.ok();
          for (std::size_t b = 0; ok && b < 4; ++b) {
            std::vector<Neighbor> hits;
            for (std::size_t j = 0; j < counts[b]; ++j) {
              hits.push_back({batch_ids[(b * kK) + j], batch_dist[(b * kK) + j]});
            }
            ok = valid(hits, removed_before);
          }
          violations += ok ? 0U : 1U;
        } else if (round % 3 == 1) {
          const auto n = f.collection->search_into(query, params, out);
          violations +=
              n.ok() && valid(std::span<const Neighbor>(out).first(n.value()), removed_before) ? 0U
                                                                                               : 1U;
        } else {
          const auto hits = f.collection->search(query, params);
          violations += hits.ok() && valid(hits.value(), removed_before) ? 0U : 1U;
        }
        // Removed ids stay removed.
        for (const ExternalId id : removed_before) {
          if (f.collection->contains(id)) {
            ++violations;
            break;
          }
        }
        ++queries;
      }
    });
  }

  for (auto& w : writers) {
    w.join();
  }
  writers_done = true;
  for (auto& r : readers) {
    r.join();
  }
  EXPECT_EQ(violations.load(), 0U);
  EXPECT_GT(queries.load(), 0U);
  EXPECT_EQ(f.collection->size(), kPreloaded - removals.load() + inserted.load());
  const vf::CollectionStats st = f.collection->stats();
  EXPECT_EQ(st.live_count, f.collection->size());
  EXPECT_EQ(st.deleted_count, removals.load());
}

TEST_P(ConcurrentSearch, CompactionWhileSearching) {
  Fixture f(GetParam());
  for (std::size_t i = 0; i < kPreloaded; i += 2) {
    ASSERT_TRUE(f.collection->remove(i).ok());
    f.removed[i].store(true);
  }
  std::atomic<bool> stop{false};
  std::atomic<std::size_t> violations{0};
  std::atomic<std::size_t> queries{0};
  const std::vector<float> query_rows = f.data.rows(9, 16);
  std::vector<std::thread> readers;
  for (std::size_t r = 0; r < 4; ++r) {
    readers.emplace_back([&, r] {
      vf::SearchParams params;
      params.k = kK;
      params.ef_search = 64;
      const std::unordered_set<ExternalId> removed_before = snapshot_removed(f);
      for (std::size_t round = 0; !stop.load() || round < 20; ++round) {
        const auto hits = f.collection->search(
            std::span<const float>(query_rows).subspan(((r + round) % 16) * kDim, kDim), params);
        violations +=
            hits.ok() && hits.value().size() == kK && valid(hits.value(), removed_before) ? 0U : 1U;
        ++queries;
      }
    });
  }
  const auto stats = f.collection->compact();
  stop = true;
  for (auto& r : readers) {
    r.join();
  }
  ASSERT_TRUE(stats.ok()) << stats.status().to_string();
  EXPECT_EQ(stats.value().rows_before, kPreloaded);
  EXPECT_EQ(stats.value().removed_rows, kPreloaded / 2);
  EXPECT_EQ(stats.value().rows_after, kPreloaded / 2);
  EXPECT_EQ(violations.load(), 0U);
  EXPECT_GT(queries.load(), 0U);
  EXPECT_EQ(f.collection->stats().deleted_count, 0U);
  EXPECT_EQ(f.collection->stats().row_count, kPreloaded / 2);
}

INSTANTIATE_TEST_SUITE_P(Indexes, ConcurrentSearch,
                         testing::Values(IndexType::Flat, IndexType::Hnsw),
                         [](const testing::TestParamInfo<IndexType>& param_info) {
                           return std::string(vf::to_string(param_info.param));
                         });

}  // namespace
