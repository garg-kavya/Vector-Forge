#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <vectorforge/types.hpp>

#include "core/rng.hpp"
#include "search/heaps.hpp"

namespace {

using vf::detail::BoundedMaxHeap;
using vf::detail::candidate_less;
using vf::detail::MinHeap;
using vf::detail::ScoredId;

enum class Pattern { Random, AllEqual, Ascending, Descending, FewDistinct };

std::vector<ScoredId> make_candidates(Pattern pattern, std::size_t n, std::uint64_t seed) {
  vf::detail::Xoshiro256ss rng(seed);
  std::vector<ScoredId> c(n);
  for (std::size_t i = 0; i < n; ++i) {
    float d = 0.0F;
    switch (pattern) {
      case Pattern::Random:
        d = vf::detail::uniform_float(rng(), -5.0F, 5.0F);
        break;
      case Pattern::AllEqual:
        d = 1.5F;
        break;
      case Pattern::Ascending:
        d = static_cast<float>(i);
        break;
      case Pattern::Descending:
        d = static_cast<float>(n - i);
        break;
      case Pattern::FewDistinct:
        d = static_cast<float>(vf::detail::uniform_below(rng, 3));
        break;
    }
    c[i] = {d, static_cast<vf::InternalId>(i)};
  }
  // Present candidates in a shuffled order so id order differs from arrival order.
  for (std::size_t i = n; i > 1; --i) {
    std::swap(c[i - 1], c[vf::detail::uniform_below(rng, i)]);
  }
  return c;
}

std::vector<ScoredId> reference_topk(std::vector<ScoredId> c, std::size_t k) {
  std::sort(c.begin(), c.end(),
            [](const ScoredId& a, const ScoredId& b) { return candidate_less(a, b); });
  c.resize(std::min(k, c.size()));
  return c;
}

TEST(CandidateOrder, LexicographicByDistanceThenId) {
  EXPECT_TRUE(candidate_less(ScoredId{1.0F, 9}, ScoredId{2.0F, 0}));
  EXPECT_TRUE(candidate_less(ScoredId{1.0F, 3}, ScoredId{1.0F, 4}));
  EXPECT_FALSE(candidate_less(ScoredId{1.0F, 4}, ScoredId{1.0F, 4}));
  EXPECT_TRUE(candidate_less(vf::Neighbor{7, -1.0F}, vf::Neighbor{2, 0.0F}));
}

TEST(TopK, HeapMatchesSortReference) {
  const Pattern patterns[] = {Pattern::Random, Pattern::AllEqual, Pattern::Ascending,
                              Pattern::Descending, Pattern::FewDistinct};
  for (const Pattern pattern : patterns) {
    for (const std::size_t n : {0U, 1U, 2U, 7U, 100U, 1000U}) {
      const std::vector<ScoredId> candidates = make_candidates(pattern, n, n + 17);
      for (const std::size_t k : {1U, 2U, 5U, 32U, 33U, 100U, 1001U}) {
        const std::vector<ScoredId> expected = reference_topk(candidates, k);

        std::vector<ScoredId> heap_storage(k);
        BoundedMaxHeap<ScoredId> heap(heap_storage);
        for (const ScoredId& c : candidates) {
          heap.push(c);
        }
        EXPECT_EQ(heap.size(), expected.size());
        const std::size_t heap_count = heap.sort_ascending();
        EXPECT_EQ(
            std::vector<ScoredId>(heap_storage.begin(),
                                  heap_storage.begin() + static_cast<std::ptrdiff_t>(heap_count)),
            expected)
            << "heap pattern=" << static_cast<int>(pattern) << " n=" << n << " k=" << k;
        EXPECT_EQ(heap.size(), 0U);
      }
    }
  }
}

TEST(TopK, PushReportsRetention) {
  std::vector<ScoredId> storage(2);
  BoundedMaxHeap<ScoredId> heap(storage);
  EXPECT_TRUE(heap.push({5.0F, 0}));
  EXPECT_TRUE(heap.push({3.0F, 1}));
  EXPECT_TRUE(heap.full());
  EXPECT_EQ(heap.top(), (ScoredId{5.0F, 0}));
  EXPECT_FALSE(heap.push({6.0F, 2}));
  EXPECT_FALSE(heap.push({5.0F, 3}));  // equal distance, larger id: not better
  EXPECT_TRUE(heap.push({4.0F, 4}));
  EXPECT_EQ(heap.top(), (ScoredId{4.0F, 4}));
  heap.pop();
  EXPECT_EQ(heap.top(), (ScoredId{3.0F, 1}));
  heap.clear();
  EXPECT_TRUE(heap.empty());
}

TEST(TopK, ZeroCapacityRetainsNothing) {
  std::vector<ScoredId> empty;
  BoundedMaxHeap<ScoredId> heap{std::span<ScoredId>(empty)};
  EXPECT_FALSE(heap.push({1.0F, 0}));
  EXPECT_EQ(heap.sort_ascending(), 0U);
}

TEST(MinHeap, PopsInAscendingOrder) {
  for (const Pattern pattern : {Pattern::Random, Pattern::FewDistinct, Pattern::Descending}) {
    const std::vector<ScoredId> candidates = make_candidates(pattern, 500, 3);
    MinHeap<ScoredId> heap;
    heap.reserve(8);
    for (const ScoredId& c : candidates) {
      heap.push(c);
    }
    EXPECT_EQ(heap.size(), candidates.size());
    const std::vector<ScoredId> expected = reference_topk(candidates, candidates.size());
    for (const ScoredId& e : expected) {
      ASSERT_FALSE(heap.empty());
      EXPECT_EQ(heap.top(), e);
      heap.pop();
    }
    EXPECT_TRUE(heap.empty());
  }
  MinHeap<ScoredId> reused;
  reused.push({2.0F, 1});
  reused.clear();
  reused.push({1.0F, 2});
  EXPECT_EQ(reused.top(), (ScoredId{1.0F, 2}));
}

}  // namespace
