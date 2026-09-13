#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "index/hnsw/neighbor_select.hpp"
#include "search/heaps.hpp"

namespace {

using vf::InternalId;
using vf::detail::NeighborSelection;
using vf::detail::ScoredId;

struct Point {
  float x;
  float y;
};

float sq_dist(Point a, Point b) {
  return ((a.x - b.x) * (a.x - b.x)) + ((a.y - b.y) * (a.y - b.y));
}

// Candidates sorted by (squared distance to base, id), as the backend provides them.
std::vector<ScoredId> sorted_candidates(Point base, const std::vector<Point>& points) {
  std::vector<ScoredId> out;
  for (std::size_t i = 0; i < points.size(); ++i) {
    out.push_back({.distance = sq_dist(base, points[i]), .id = static_cast<InternalId>(i)});
  }
  std::sort(out.begin(), out.end(),
            [](const ScoredId& a, const ScoredId& b) { return vf::detail::candidate_less(a, b); });
  return out;
}

std::vector<InternalId> select(const std::vector<Point>& points, Point base, std::size_t m,
                               NeighborSelection selection, bool keep_pruned) {
  const std::vector<ScoredId> candidates = sorted_candidates(base, points);
  std::vector<ScoredId> out;
  std::vector<ScoredId> discarded;
  vf::detail::select_neighbors(
      candidates, m, selection, keep_pruned,
      [&](InternalId a, InternalId b) { return sq_dist(points[a], points[b]); }, out, discarded);
  std::vector<InternalId> ids;
  for (const ScoredId& s : out) {
    ids.push_back(s.id);
  }
  return ids;
}

// Base at the origin. 0 = A(1, 0); 1 = B(1.1, 0.1), right behind A; 2 = C(0, 1.2); 3 = D(-1.3, 0).
const std::vector<Point> kPoints = {{1.0F, 0.0F}, {1.1F, 0.1F}, {0.0F, 1.2F}, {-1.3F, 0.0F}};
constexpr Point kOrigin{0.0F, 0.0F};

TEST(NeighborSelect, HeuristicPrunesCandidatesShadowedBySelectedNeighbours) {
  // B is much closer to A than to the base, so it is pruned; C and D point in other directions.
  EXPECT_EQ(select(kPoints, kOrigin, 3, NeighborSelection::Heuristic, false),
            (std::vector<InternalId>{0, 2, 3}));
  EXPECT_EQ(select(kPoints, kOrigin, 4, NeighborSelection::Heuristic, false),
            (std::vector<InternalId>{0, 2, 3}))
      << "without keep_pruned the selection may stay below m";
}

TEST(NeighborSelect, KeepPrunedRefillsInAscendingOrder) {
  EXPECT_EQ(select(kPoints, kOrigin, 4, NeighborSelection::Heuristic, true),
            (std::vector<InternalId>{0, 2, 3, 1}));
  EXPECT_EQ(select(kPoints, kOrigin, 3, NeighborSelection::Heuristic, true),
            (std::vector<InternalId>{0, 2, 3}))
      << "refill only happens when slots remain";
}

TEST(NeighborSelect, StopsAtM) {
  EXPECT_EQ(select(kPoints, kOrigin, 2, NeighborSelection::Heuristic, true),
            (std::vector<InternalId>{0, 2}));
  EXPECT_EQ(select(kPoints, kOrigin, 1, NeighborSelection::Heuristic, false),
            (std::vector<InternalId>{0}));
}

TEST(NeighborSelect, SimpleTakesClosest) {
  EXPECT_EQ(select(kPoints, kOrigin, 3, NeighborSelection::Simple, false),
            (std::vector<InternalId>{0, 1, 2}));
  EXPECT_EQ(select(kPoints, kOrigin, 10, NeighborSelection::Simple, false),
            (std::vector<InternalId>{0, 1, 2, 3}));
}

TEST(NeighborSelect, EmptyInput) {
  EXPECT_TRUE(select({}, kOrigin, 4, NeighborSelection::Heuristic, true).empty());
  EXPECT_TRUE(select({}, kOrigin, 4, NeighborSelection::Simple, true).empty());
}

TEST(NeighborSelect, DuplicateCandidates) {
  // Three identical points: once one is selected the others are at distance 0 from it, closer than
  // to the base, so the heuristic keeps a single one unless keep_pruned refills.
  const std::vector<Point> dups = {{1.0F, 0.0F}, {1.0F, 0.0F}, {1.0F, 0.0F}};
  EXPECT_EQ(select(dups, kOrigin, 3, NeighborSelection::Heuristic, false),
            (std::vector<InternalId>{0}));
  EXPECT_EQ(select(dups, kOrigin, 3, NeighborSelection::Heuristic, true),
            (std::vector<InternalId>{0, 1, 2}));
  // Candidates identical to the base: 0 < 0 is false, so none shadows another.
  EXPECT_EQ(select(dups, Point{1.0F, 0.0F}, 3, NeighborSelection::Heuristic, false),
            (std::vector<InternalId>{0, 1, 2}));
}

TEST(NeighborSelect, DoesNotAllocateWithReservedCapacity) {
  const std::vector<ScoredId> candidates = sorted_candidates(kOrigin, kPoints);
  std::vector<ScoredId> out;
  std::vector<ScoredId> discarded;
  out.reserve(candidates.size());
  discarded.reserve(candidates.size());
  const auto* out_data = out.data();
  const auto* discarded_data = discarded.data();
  vf::detail::select_neighbors(
      candidates, 4, NeighborSelection::Heuristic, true,
      [&](InternalId a, InternalId b) { return sq_dist(kPoints[a], kPoints[b]); }, out, discarded);
  EXPECT_EQ(out.data(), out_data);
  EXPECT_EQ(discarded.data(), discarded_data);
  EXPECT_EQ(out.size(), 4U);
}

}  // namespace
