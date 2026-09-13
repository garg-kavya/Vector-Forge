#pragma once

// Distance from a prepared query to one stored row, under the collection's metric and
// normalisation (the per-pair counterpart of FlatBackend::block_distances).
//
// For normalised collections stored rows have unit norm and q' = q * inv_norm:
//   Cosine: max(0, 1 - <q', x>)   L2: max(0, 2 - 2<q', x>)   InnerProduct: -<q', x>
// A stored row used as the query has inv_norm = 1, and every formula is symmetric bit for bit
// (dot and squared-difference sums do not depend on argument order), so dist(a, b) == dist(b, a).

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <vectorforge/types.hpp>

#include "index/index_backend.hpp"
#include "simd/kernels.hpp"

namespace vf::detail {

enum class ScoreMode : std::uint8_t {
  L2,             // squared L2 of raw vectors
  NegDot,         // -dot of raw vectors (inner product)
  NormalizedCos,  // unit rows, cosine distance
  NormalizedL2,   // unit rows, squared L2 via dot
  NormalizedIp,   // unit rows, -dot
};

[[nodiscard]] constexpr ScoreMode score_mode(Metric metric, bool normalized) noexcept {
  if (!normalized) {
    return metric == Metric::L2 ? ScoreMode::L2 : ScoreMode::NegDot;
  }
  switch (metric) {
    case Metric::L2:
      return ScoreMode::NormalizedL2;
    case Metric::InnerProduct:
      return ScoreMode::NormalizedIp;
    case Metric::Cosine:
      return ScoreMode::NormalizedCos;
  }
  return ScoreMode::NormalizedCos;
}

[[nodiscard]] inline float query_distance(const KernelTable& kernels, ScoreMode mode,
                                          const QueryView& query, const float* row,
                                          std::size_t dim) noexcept {
  switch (mode) {
    case ScoreMode::L2:
      return kernels.l2sq(query.data, row, dim);
    case ScoreMode::NegDot:
      return -kernels.dot(query.data, row, dim);
    case ScoreMode::NormalizedCos:
      return std::max(0.0F, 1.0F - (kernels.dot(query.data, row, dim) * query.inv_norm));
    case ScoreMode::NormalizedL2:
      return std::max(0.0F, 2.0F - (2.0F * kernels.dot(query.data, row, dim) * query.inv_norm));
    case ScoreMode::NormalizedIp:
      return -(kernels.dot(query.data, row, dim) * query.inv_norm);
  }
  return kernels.l2sq(query.data, row, dim);
}

}  // namespace vf::detail
