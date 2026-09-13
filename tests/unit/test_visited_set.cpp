#include <gtest/gtest.h>

#include <cstdint>

#include "search/context_pool.hpp"
#include "search/visited_set.hpp"

namespace {

using vf::detail::ContextPool;
using vf::detail::VisitedSet;

TEST(VisitedSet, VisitOncePerSearch) {
  VisitedSet set;
  set.ensure_capacity(10);
  EXPECT_EQ(set.capacity(), 10U);
  EXPECT_TRUE(set.visit(3));
  EXPECT_FALSE(set.visit(3));
  EXPECT_TRUE(set.visited(3));
  EXPECT_FALSE(set.visited(4));
  EXPECT_FALSE(set.visited(100)) << "out-of-range ids are reported unvisited";
  set.reset();
  EXPECT_FALSE(set.visited(3));
  EXPECT_TRUE(set.visit(3));
}

TEST(VisitedSet, GrowthKeepsNewIdsUnvisited) {
  VisitedSet set;
  set.ensure_capacity(4);
  EXPECT_TRUE(set.visit(1));
  set.ensure_capacity(1000);
  EXPECT_TRUE(set.visited(1)) << "growing does not reset the current search";
  EXPECT_TRUE(set.visit(999));
  set.ensure_capacity(8);  // never shrinks
  EXPECT_EQ(set.capacity(), 1000U);
}

TEST(VisitedSet, EpochWrapAroundClearsMarks) {
  VisitedSet set;
  set.ensure_capacity(3);
  // Mark id 0 in a search whose epoch will be reused after the wrap-around.
  EXPECT_EQ(set.epoch(), 1U);
  EXPECT_TRUE(set.visit(0));
  for (int i = 0; i < 65534; ++i) {
    set.reset();
  }
  EXPECT_EQ(set.epoch(), 65535U);
  EXPECT_TRUE(set.visit(1));
  set.reset();  // wraps: clears and restarts at epoch 1
  EXPECT_EQ(set.epoch(), 1U);
  EXPECT_FALSE(set.visited(0)) << "a stale mark equal to the new epoch must have been cleared";
  EXPECT_FALSE(set.visited(1));
  EXPECT_TRUE(set.visit(0));
  EXPECT_TRUE(set.visit(1));
  EXPECT_TRUE(set.visit(2));
}

TEST(ContextPool, ReusesReleasedContexts) {
  ContextPool pool;
  vf::detail::SearchContext* first = nullptr;
  {
    const ContextPool::Lease lease = pool.acquire();
    first = &*lease;
    lease->distance_computations = 5;
  }
  EXPECT_EQ(pool.created(), 1U);
  {
    const ContextPool::Lease a = pool.acquire();
    EXPECT_EQ(&*a, first);
    EXPECT_EQ(a->distance_computations, 5U);
    const ContextPool::Lease b = pool.acquire();  // a is still leased
    EXPECT_NE(&*b, first);
  }
  EXPECT_EQ(pool.created(), 2U);
  const ContextPool::Lease c = pool.acquire();
  const ContextPool::Lease d = pool.acquire();
  EXPECT_EQ(pool.created(), 2U) << "both idle contexts are reused";
}

TEST(SearchContext, PrepareSizesStorage) {
  vf::detail::SearchContext context;
  context.prepare(100, 32);
  EXPECT_GE(context.visited.capacity(), 100U);
  EXPECT_GE(context.results.size(), 32U);
  context.prepare(10, 8);  // never shrinks
  EXPECT_GE(context.visited.capacity(), 100U);
  EXPECT_GE(context.results.size(), 32U);
  EXPECT_GT(context.bytes(), 0U);
}

}  // namespace
