// Exception safety of Collection writes under injected allocation failures (docs/DESIGN.md §4.7).
//
// For every insert, each allocation the insert performs is made to fail in turn. A failed insert
// must leave the collection observably unchanged (strong guarantee), and the collection must stay
// fully usable: afterwards every live vector is still found by an exact-match search.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <new>
#include <span>
#include <string>
#include <vector>

#include <vectorforge/collection.hpp>

#include "core/rng.hpp"
#include "counting_allocator.hpp"
#include "support/test_data.hpp"

namespace {

using vf::Collection;
using vf::ExternalId;
using vf::IndexType;

constexpr std::uint32_t kDim = 8;

struct Snapshot {
  std::size_t size;
  std::uint64_t rows;
  std::uint64_t deleted;

  friend bool operator==(const Snapshot&, const Snapshot&) = default;
};

Snapshot snapshot(const Collection& c) {
  const vf::CollectionStats st = c.stats();
  return {c.size(), st.row_count, st.deleted_count};
}

std::unique_ptr<Collection> make(IndexType index) {
  vf::CollectionConfig cfg;
  cfg.dim = kDim;
  cfg.index = index;
  cfg.hnsw.M = 6;
  cfg.hnsw.ef_construction = 24;
  return Collection::create(cfg).value();
}

// Every live vector must be its own nearest neighbour (all test vectors are distinct).
void expect_all_findable(const Collection& c,
                         const std::map<ExternalId, std::vector<float>>& model) {
  vf::SearchParams p;
  p.k = 1;
  p.ef_search = 256;
  std::size_t missing = 0;
  for (const auto& [id, v] : model) {
    ASSERT_EQ(c.get(id).value(), v) << "id " << id;
    const auto hits = c.search(v, p).value();
    if (hits.size() != 1 || hits[0].id != id || hits[0].distance != 0.0F) {
      ++missing;
    }
  }
  EXPECT_EQ(missing, 0U);
}

class ExceptionSafety : public testing::TestWithParam<IndexType> {
 protected:
  void SetUp() override {
#if defined(_MSC_VER) && defined(_ITERATOR_DEBUG_LEVEL) && _ITERATOR_DEBUG_LEVEL == 2
    // The MSVC debug STL allocates a container proxy inside noexcept constructors (for example
    // std::vector's default constructor), so an injected failure there calls std::terminate
    // regardless of the code under test. The test runs in every other configuration.
    GTEST_SKIP() << "allocation failure injection is incompatible with _ITERATOR_DEBUG_LEVEL=2";
#endif
  }
};

TEST_P(ExceptionSafety, AddIsStrongUnderAllocationFailure) {
  const std::unique_ptr<Collection> c = make(GetParam());
  std::map<ExternalId, std::vector<float>> model;
  vf::detail::Xoshiro256ss rng(2024);
  std::size_t injected = 0;
  constexpr ExternalId kInserts = 400;

  for (ExternalId step = 0; step < kInserts; ++step) {
    // Every 4th step replaces an existing id (upsert) instead of inserting a new one.
    const bool upsert = step % 4 == 3;
    const ExternalId id = upsert ? step - 2 : step;
    const std::vector<float> v = vf::test::random_vector(rng, kDim);
    for (std::size_t fail_at = 0;; ++fail_at) {
      const Snapshot before = snapshot(*c);
      bool threw = false;
      bool triggered = false;
      vf::Status st;
      {
        const vf::test::AllocationFailure failure(fail_at);
        try {
          st = c->add(id, v, {.upsert = upsert});
        } catch (const std::bad_alloc&) {
          threw = true;
        }
        triggered = failure.triggered();
      }
      if (threw) {
        ++injected;
        ASSERT_EQ(snapshot(*c), before) << "step " << step << " fail_at " << fail_at;
        ASSERT_EQ(c->contains(id), model.contains(id));
        if (model.contains(id)) {
          ASSERT_EQ(c->get(id).value(), model[id]) << "a failed upsert must keep the old vector";
        }
        continue;
      }
      ASSERT_TRUE(st.ok()) << st.to_string();
      model[id] = v;
      ASSERT_EQ(c->size(), model.size());
      if (!triggered) {
        break;  // the insert completed without reaching the injected failure: all points covered
      }
    }
  }
  EXPECT_GT(injected, kInserts / 2) << "the test must actually inject failures";
  expect_all_findable(*c, model);
}

TEST_P(ExceptionSafety, AddBatchRollsBackOnlyTheFailingRow) {
  const std::unique_ptr<Collection> c = make(GetParam());
  std::map<ExternalId, std::vector<float>> model;
  vf::detail::Xoshiro256ss rng(77);
  constexpr std::size_t kBatch = 5;
  for (std::size_t batch = 0; batch < 40; ++batch) {
    std::vector<ExternalId> ids(kBatch);
    for (std::size_t i = 0; i < kBatch; ++i) {
      ids[i] = (batch * kBatch) + i;
    }
    const std::vector<float> rows = vf::test::random_matrix(rng, kBatch, kDim);
    for (std::size_t fail_at = 0;; ++fail_at) {
      bool threw = false;
      bool triggered = false;
      vf::Status batch_status;
      {
        const vf::test::AllocationFailure failure(fail_at);
        try {
          batch_status = c->add_batch(ids, rows, {.upsert = true}).status();
        } catch (const std::bad_alloc&) {
          threw = true;
        }
        triggered = failure.triggered();
      }
      // Rows before the failure stay inserted (documented); the failing row leaves no trace.
      for (std::size_t i = 0; i < kBatch; ++i) {
        if (c->contains(ids[i])) {
          model[ids[i]] =
              std::vector<float>(rows.begin() + static_cast<std::ptrdiff_t>(i * kDim),
                                 rows.begin() + static_cast<std::ptrdiff_t>((i + 1) * kDim));
        }
      }
      const vf::CollectionStats st = c->stats();
      ASSERT_EQ(st.live_count, model.size());
      ASSERT_EQ(st.row_count, st.live_count + st.deleted_count);
      if (!threw) {
        ASSERT_TRUE(batch_status.ok()) << batch_status.to_string();
        if (!triggered) {
          break;
        }
      }
    }
  }
  expect_all_findable(*c, model);
}

INSTANTIATE_TEST_SUITE_P(Indexes, ExceptionSafety,
                         testing::Values(IndexType::Flat, IndexType::Hnsw),
                         [](const testing::TestParamInfo<IndexType>& param_info) {
                           return std::string(vf::to_string(param_info.param));
                         });

}  // namespace
