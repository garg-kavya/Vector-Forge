#pragma once

// Independent double-precision brute-force reference for exact k-NN (docs/DESIGN.md §15.2).
//
// The reference shares no code with the library: it evaluates distances in double precision
// directly from the *raw* inserted vectors (normalising in double where the collection normalises)
// and sorts everything by (distance, insertion order).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

#include <vectorforge/types.hpp>

namespace vf::test {

struct RefItem {
  ExternalId id = 0;
  std::uint64_t order = 0;  // insertion sequence number (matches internal id order)
  std::span<const float> vector;
};

struct RefScored {
  ExternalId id = 0;
  std::uint64_t order = 0;
  double distance = 0.0;
};

inline double ref_norm(std::span<const float> v) {
  double s = 0.0;
  for (float x : v) {
    s += static_cast<double>(x) * static_cast<double>(x);
  }
  return std::sqrt(s);
}

// Lower-is-better distance under VectorForge's convention.
inline double ref_distance(Metric metric, bool normalized, std::span<const float> query,
                           std::span<const float> stored) {
  const double qn = normalized ? ref_norm(query) : 1.0;
  const double xn = normalized ? ref_norm(stored) : 1.0;
  double dot = 0.0;
  double l2 = 0.0;
  for (std::size_t i = 0; i < query.size(); ++i) {
    const double q = static_cast<double>(query[i]) / qn;
    const double x = static_cast<double>(stored[i]) / xn;
    dot += q * x;
    l2 += (q - x) * (q - x);
  }
  switch (metric) {
    case Metric::L2:
      return l2;
    case Metric::InnerProduct:
      return -dot;
    case Metric::Cosine:
      return 1.0 - dot;
  }
  return l2;
}

inline std::vector<RefScored> reference_ranking(Metric metric, bool normalized,
                                                std::span<const float> query,
                                                const std::vector<RefItem>& items) {
  std::vector<RefScored> out;
  out.reserve(items.size());
  for (const RefItem& item : items) {
    out.push_back({item.id, item.order, ref_distance(metric, normalized, query, item.vector)});
  }
  std::sort(out.begin(), out.end(), [](const RefScored& a, const RefScored& b) {
    return a.distance < b.distance || (a.distance == b.distance && a.order < b.order);
  });
  return out;
}

// Exact comparison: same ids in the same order, float distances equal to the reference distances.
// Valid when float arithmetic is exact (see integer_vector()).
inline void expect_exact_topk(std::span<const Neighbor> actual,
                              const std::vector<RefScored>& ranking, std::size_t k) {
  const std::size_t expected = std::min(k, ranking.size());
  ASSERT_EQ(actual.size(), expected);
  for (std::size_t i = 0; i < expected; ++i) {
    EXPECT_EQ(actual[i].id, ranking[i].id) << "rank " << i;
    EXPECT_EQ(actual[i].distance, static_cast<float>(ranking[i].distance)) << "rank " << i;
  }
}

// Tie-tolerant comparison for inexact arithmetic: distances match the reference within `tol`,
// results are sorted, ids are unique, nothing clearly better than the k-th reference distance is
// missing and nothing clearly worse is included.
inline void expect_tolerant_topk(std::span<const Neighbor> actual,
                                 const std::vector<RefScored>& ranking, std::size_t k, double tol) {
  const std::size_t expected = std::min(k, ranking.size());
  ASSERT_EQ(actual.size(), expected);
  if (expected == 0) {
    return;
  }
  std::unordered_map<ExternalId, double> ref_by_id;
  for (const RefScored& r : ranking) {
    ref_by_id.emplace(r.id, r.distance);
  }
  const double kth = ranking[expected - 1].distance;
  std::unordered_map<ExternalId, int> seen;
  for (std::size_t i = 0; i < expected; ++i) {
    const auto it = ref_by_id.find(actual[i].id);
    ASSERT_NE(it, ref_by_id.end()) << "unknown id " << actual[i].id;
    EXPECT_NEAR(static_cast<double>(actual[i].distance), it->second, tol) << "id " << actual[i].id;
    EXPECT_LE(it->second, kth + (2 * tol)) << "id " << actual[i].id << " should not be in top-k";
    EXPECT_EQ(++seen[actual[i].id], 1) << "duplicate id " << actual[i].id;
    if (i > 0) {
      EXPECT_LE(actual[i - 1].distance, actual[i].distance) << "unsorted at rank " << i;
    }
  }
  for (const RefScored& r : ranking) {
    if (r.distance < kth - (2 * tol)) {
      EXPECT_TRUE(seen.contains(r.id))
          << "missing id " << r.id << " (distance " << r.distance << ")";
    }
  }
}

}  // namespace vf::test
