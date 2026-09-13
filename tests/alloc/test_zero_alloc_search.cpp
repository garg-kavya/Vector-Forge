// Verifies that Collection::search_into performs no heap allocation (docs/DESIGN.md §4.5).

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <vector>

#include <vectorforge/collection.hpp>

#include "core/rng.hpp"
#include "counting_allocator.hpp"
#include "support/test_data.hpp"

namespace {

using vf::Collection;
using vf::Metric;
using vf::Neighbor;

TEST(AllocationCounter, DetectsAllocations) {
  std::size_t counted = 0;
  {
    const vf::test::AllocationCounter counter;
    // Direct calls of the allocation functions (unlike new-expressions) may not be elided.
    void* p = ::operator new(64);
    void* q = ::operator new(64, std::align_val_t{64});
    counted = counter.count();
    ::operator delete(q, std::align_val_t{64});
    ::operator delete(p);
  }
  EXPECT_EQ(counted, 2U) << "the counting allocator must observe allocations";
}

struct Case {
  Metric metric;
  bool normalize;
  std::uint32_t k;
};

TEST(ZeroAllocation, SearchIntoDoesNotAllocate) {
  constexpr std::uint32_t kDim = 32;
  const Case cases[] = {{Metric::L2, false, 10},
                        {Metric::InnerProduct, false, 64},
                        {Metric::Cosine, false, 5},
                        {Metric::L2, true, 100},
                        {Metric::L2, false, 5000}};
  for (const Case& c : cases) {
    SCOPED_TRACE(testing::Message()
                 << vf::to_string(c.metric) << " normalize=" << c.normalize << " k=" << c.k);
    vf::CollectionConfig cfg;
    cfg.dim = kDim;
    cfg.metric = c.metric;
    cfg.normalize = c.normalize;
    cfg.index = vf::IndexType::Flat;
    const std::unique_ptr<Collection> col = Collection::create(cfg).value();
    vf::detail::Xoshiro256ss rng(7);
    for (vf::ExternalId id = 0; id < 3000; ++id) {
      ASSERT_TRUE(col->add(id, vf::test::random_vector(rng, kDim)).ok());
    }
    for (vf::ExternalId id = 0; id < 3000; id += 5) {
      ASSERT_TRUE(col->remove(id).ok());  // exercise the tombstone path too
    }
    const std::vector<float> queries = vf::test::random_matrix(rng, 50, kDim);
    std::vector<Neighbor> out(c.k);
    vf::SearchParams params;
    params.k = c.k;

    std::size_t total_results = 0;
    std::size_t failures = 0;
    std::size_t allocations = 0;
    {
      const vf::test::AllocationCounter counter;
      for (std::size_t q = 0; q < 50; ++q) {
        const vf::Result<std::size_t> n =
            col->search_into(std::span<const float>(queries).subspan(q * kDim, kDim), params, out);
        if (n.ok()) {
          total_results += n.value();
        } else {
          ++failures;
        }
      }
      allocations = counter.count();
    }
    EXPECT_EQ(failures, 0U);
    EXPECT_GT(total_results, 0U);
    EXPECT_EQ(allocations, 0U);
  }
}

}  // namespace
