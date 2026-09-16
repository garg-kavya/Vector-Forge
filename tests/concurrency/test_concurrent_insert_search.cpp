// Level B (docs/DESIGN.md §11.4, §11.7; docs/concurrency.md "Level B"): searches, removals and
// saves run while a parallel add_batch links rows. Checked:
//   - every result is sorted, duplicate-free and names only ids that were inserted;
//   - deletion is linearised at remove(): once remove(id) has returned, no search that starts
//     afterwards returns id (a search running concurrently with remove may or may not);
//   - an id whose add_batch has returned is found by an exact-match search in the vast majority
//     of cases (HNSW is approximate, so this is a rate, not a guarantee);
//   - snapshots saved during ingestion load with full verification and pass the graph validator.
// Run under TSan (linux-clang-tsan); the nightly workflow repeats the label 100 times.

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <unordered_set>
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
using vf::Neighbor;

constexpr std::uint32_t kDim = 8;
constexpr std::size_t kPreloaded = 1000;
constexpr std::size_t kBatches = 12;
constexpr std::size_t kBatchRows = 400;
constexpr std::uint32_t kK = 10;

std::unique_ptr<Collection> make() {
  vf::CollectionConfig cfg;
  cfg.dim = kDim;
  cfg.index = vf::IndexType::Hnsw;
  cfg.concurrency = vf::Concurrency::Concurrent;
  cfg.hnsw.M = 8;
  cfg.hnsw.ef_construction = 48;
  return Collection::create(cfg).value();
}

TEST(ConcurrentInsertSearch, SearchesRemovalsAndSavesDuringParallelIngest) {
  const std::unique_ptr<Collection> c = make();
  const vf::test::ClusteredData data(21, kDim, 16, 0.3F);
  const std::vector<float> preload = data.rows(1, kPreloaded);
  std::vector<ExternalId> preload_ids(kPreloaded);
  for (std::size_t i = 0; i < kPreloaded; ++i) {
    preload_ids[i] = i;
  }
  ASSERT_EQ(c->add_batch(preload_ids, preload).value(), kPreloaded);

  const std::vector<float> ingest = data.rows(2, kBatches * kBatchRows);
  constexpr ExternalId kIngestBase = 100000;
  std::vector<std::atomic<bool>> removed(kPreloaded);
  std::atomic<std::size_t> batches_done{0};
  std::atomic<bool> writer_done{false};
  // Violations by kind, so that a failure names what went wrong.
  enum Kind : std::size_t { kWriter, kRemover, kSave, kLoad, kValidate, kSearch, kResult, kKinds };
  constexpr std::array<const char*, kKinds> kKindNames{"writer",   "remover", "save",  "load",
                                                       "validate", "search",  "result"};
  std::array<std::atomic<std::size_t>, kKinds> violations{};
  std::atomic<std::size_t> queries{0};
  std::atomic<std::size_t> exact_found{0};
  std::atomic<std::size_t> exact_checked{0};

  std::thread writer([&] {
    vf::ThreadPool pool(3);
    for (std::size_t b = 0; b < kBatches; ++b) {
      std::vector<ExternalId> ids(kBatchRows);
      for (std::size_t i = 0; i < kBatchRows; ++i) {
        ids[i] = kIngestBase + (b * kBatchRows) + i;
      }
      const auto rows =
          std::span<const float>(ingest).subspan(b * kBatchRows * kDim, kBatchRows * kDim);
      const auto added = c->add_batch(ids, rows, {}, &pool);
      violations[kWriter] += added.ok() && added.value() == kBatchRows ? 0U : 1U;
      batches_done.store(b + 1);
    }
    writer_done.store(true);
  });

  std::thread remover([&] {
    for (std::size_t i = 0; i < kPreloaded && !writer_done.load(); i += 3) {
      violations[kRemover] += c->remove(i).ok() ? 0U : 1U;
      removed[i].store(true);
      std::this_thread::yield();
    }
  });

  const vf::test::ScopedTempDir dir("insert_search");
  std::thread saver([&] {
    for (int round = 0; round < 3 && !writer_done.load(); ++round) {
      const std::filesystem::path file = dir.file("snap" + std::to_string(round) + ".vfidx");
      violations[kSave] += c->save(file).ok() ? 0U : 1U;
      const auto loaded = Collection::load(file, {.use_mmap = false, .verify = vf::Verify::Full});
      if (!loaded.ok()) {
        ++violations[kLoad];
        ADD_FAILURE() << loaded.status().to_string();
        continue;
      }
      const auto& backend = static_cast<const vf::detail::HnswBackend&>(
          *vf::detail::CollectionFactory::state(*loaded.value()).backend);
      const vf::Status valid = vf::detail::HnswValidator(backend.graph()).check_invariants();
      violations[kValidate] += valid.ok() ? 0U : 1U;
    }
  });

  std::vector<std::thread> readers;
  for (std::size_t r = 0; r < 3; ++r) {
    readers.emplace_back([&, r] {
      vf::SearchParams params;
      params.k = kK;
      params.ef_search = 48;
      std::vector<Neighbor> out(kK);
      for (std::size_t round = 0; !writer_done.load() || round < 30; ++round) {
        std::unordered_set<ExternalId> removed_before;
        for (std::size_t i = 0; i < kPreloaded; i += 3) {
          if (removed[i].load()) {
            removed_before.insert(i);
          }
        }
        const std::size_t done = batches_done.load();
        // Alternate between a preloaded vector and a vector of an already inserted batch.
        std::span<const float> query;
        ExternalId expected = vf::kInvalidExternalId;
        if (round % 2 == 1 && done > 0) {
          const std::size_t row = ((r * 7919) + (round * 104729)) % (done * kBatchRows);
          query = std::span<const float>(ingest).subspan(row * kDim, kDim);
          expected = kIngestBase + row;
        } else {
          const std::size_t row = ((r * 31) + round) % kPreloaded;
          query = std::span<const float>(preload).subspan(row * kDim, kDim);
        }
        const auto n = c->search_into(query, params, out);
        if (!n.ok()) {
          ++violations[kSearch];
          continue;
        }
        std::unordered_set<ExternalId> seen;
        bool ok = true;
        bool hit = false;
        for (std::size_t i = 0; i < n.value(); ++i) {
          const ExternalId id = out[i].id;
          const bool known =
              id < kPreloaded || (id >= kIngestBase && id < kIngestBase + (kBatches * kBatchRows));
          ok = ok && known && !removed_before.contains(id) && seen.insert(id).second &&
               (i == 0 || out[i - 1].distance <= out[i].distance);
          hit = hit || (id == expected && out[i].distance == 0.0F);
        }
        violations[kResult] += ok ? 0U : 1U;
        if (expected != vf::kInvalidExternalId) {
          ++exact_checked;
          exact_found += hit ? 1U : 0U;
        }
        ++queries;
      }
    });
  }

  writer.join();
  remover.join();
  saver.join();
  for (auto& t : readers) {
    t.join();
  }
  for (std::size_t kind = 0; kind < kKinds; ++kind) {
    EXPECT_EQ(violations[kind].load(), 0U) << kKindNames[kind];
  }
  EXPECT_GT(queries.load(), 0U);
  if (exact_checked.load() > 0) {
    EXPECT_GE(exact_found.load() * 100, exact_checked.load() * 95)
        << exact_found.load() << " of " << exact_checked.load();
  }

  std::size_t removed_count = 0;
  for (const auto& flag : removed) {
    removed_count += flag.load() ? 1U : 0U;
  }
  EXPECT_EQ(c->size(), kPreloaded - removed_count + (kBatches * kBatchRows));
  const auto& backend = static_cast<const vf::detail::HnswBackend&>(
      *vf::detail::CollectionFactory::state(*c).backend);
  const vf::detail::HnswValidator validator(backend.graph());
  ASSERT_TRUE(validator.check_invariants().ok());
  EXPECT_EQ(validator.reachability().unreachable_level0(), 0U);
}

}  // namespace
