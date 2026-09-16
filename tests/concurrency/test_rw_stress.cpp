// Mixed read/write stress (label "stress"; nightly CI repeats it 100 times, docs/concurrency.md).
// Readers, a writer, a compactor and a saver run concurrently on one collection. The final state
// must equal a model of the writer's operations, and every search result must be well-formed.

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include <vectorforge/collection.hpp>
#include <vectorforge/thread_pool.hpp>

#include "core/rng.hpp"
#include "support/test_data.hpp"

namespace {

using vf::Collection;
using vf::ExternalId;
using vf::IndexType;

constexpr std::uint32_t kDim = 6;

class RwStress : public testing::TestWithParam<IndexType> {};

TEST_P(RwStress, FinalStateMatchesModel) {
  vf::CollectionConfig cfg;
  cfg.dim = kDim;
  cfg.metric = vf::Metric::Cosine;
  cfg.index = GetParam();
  cfg.hnsw.M = 6;
  cfg.hnsw.ef_construction = 32;
  const std::unique_ptr<Collection> c = Collection::create(cfg).value();
  const vf::test::ScopedTempDir dir("rw_stress");
  vf::ThreadPool pool(2);

  std::atomic<bool> done{false};
  std::atomic<std::size_t> bad{0};
  std::map<ExternalId, std::vector<float>> model;  // owned by the writer thread until joined

  std::thread writer([&] {
    vf::detail::Xoshiro256ss rng(2026);
    for (std::size_t op = 0; op < 1500; ++op) {
      const ExternalId id = vf::detail::uniform_below(rng, 400);
      const std::uint64_t kind = vf::detail::uniform_below(rng, 10);
      if (kind < 6) {
        std::vector<float> v = vf::test::random_vector(rng, kDim, 0.1F, 1.0F);
        const vf::Status st = c->add(id, v, {.upsert = true});
        if (!st.ok()) {
          ++bad;
          continue;
        }
        // The model keeps the stored (normalised) form.
        model[id] = c->get(id).value();
      } else if (kind < 9) {
        const vf::Status st = c->remove(id);
        const bool expected = model.erase(id) == 1;
        bad += st.ok() == expected ? 0U : 1U;
      } else {
        std::vector<ExternalId> ids;
        std::vector<float> rows;
        for (ExternalId b = 0; b < 8; ++b) {
          ids.push_back(400 + ((id + b) % 200));
          const std::vector<float> v = vf::test::random_vector(rng, kDim, 0.1F, 1.0F);
          rows.insert(rows.end(), v.begin(), v.end());
        }
        const auto added = c->add_batch(ids, rows, {.upsert = true}, &pool);
        bad += added.ok() && added.value() == ids.size() ? 0U : 1U;
        for (const ExternalId bid : ids) {
          model[bid] = c->get(bid).value();
        }
      }
    }
    done = true;
  });

  std::thread compactor([&] {
    while (!done.load()) {
      bad += c->compact().ok() ? 0U : 1U;
      std::this_thread::yield();
    }
  });

  std::thread saver([&] {
    std::size_t round = 0;
    while (!done.load()) {
      const std::filesystem::path file =
          dir.path() / ("snap" + std::to_string(round % 2) + ".vfidx");
      bad += c->save(file).ok() ? 0U : 1U;
      ++round;
    }
  });

  std::vector<std::thread> readers;
  for (std::size_t r = 0; r < 3; ++r) {
    readers.emplace_back([&, r] {
      vf::detail::Xoshiro256ss rng(r + 1);
      vf::SearchParams params;
      params.k = 5;
      while (!done.load()) {
        const std::vector<float> q = vf::test::random_vector(rng, kDim, 0.1F, 1.0F);
        const auto hits = c->search(q, params);
        if (!hits.ok() || hits.value().size() > 5 ||
            !std::is_sorted(hits.value().begin(), hits.value().end(),
                            [](const vf::Neighbor& a, const vf::Neighbor& b) {
                              return a.distance < b.distance;
                            })) {
          ++bad;
        }
        static_cast<void>(c->stats());
      }
    });
  }

  writer.join();
  compactor.join();
  saver.join();
  for (auto& r : readers) {
    r.join();
  }
  EXPECT_EQ(bad.load(), 0U);
  ASSERT_EQ(c->size(), model.size());
  for (const auto& [id, v] : model) {
    const auto stored = c->get(id);
    ASSERT_TRUE(stored.ok()) << id;
    EXPECT_EQ(stored.value(), v) << id;
  }
  // A final compaction leaves exactly the live rows; the saved snapshot loads.
  ASSERT_TRUE(c->compact().ok());
  EXPECT_EQ(c->stats().row_count, model.size());
  ASSERT_TRUE(c->save(dir.path() / "final.vfidx").ok());
  const auto loaded = Collection::load(dir.path() / "final.vfidx");
  ASSERT_TRUE(loaded.ok()) << loaded.status().to_string();
  EXPECT_EQ(loaded.value()->size(), model.size());
}

INSTANTIATE_TEST_SUITE_P(Indexes, RwStress, testing::Values(IndexType::Flat, IndexType::Hnsw),
                         [](const testing::TestParamInfo<IndexType>& param_info) {
                           return std::string(vf::to_string(param_info.param));
                         });

}  // namespace
