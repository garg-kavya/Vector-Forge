#pragma once

// Metric adapter over the kernel table. Kept out of kernels.hpp so that translation units compiled
// with special floating-point or instruction-set flags contain no shared inline code.

#include <cstddef>

#include <vectorforge/types.hpp>

#include "simd/kernels.hpp"

namespace vf::detail {

// Lower-is-better distance for `metric` under the internal convention: squared L2, -dot, or
// 1 - dot (vectors already normalised for Metric::Cosine). Precondition: is_valid(metric).
[[nodiscard]] inline float metric_distance(const KernelTable& table, Metric metric, const float* a,
                                           const float* b, std::size_t dim) noexcept {
  switch (metric) {
    case Metric::L2:
      return table.l2sq(a, b, dim);
    case Metric::InnerProduct:
      return -table.dot(a, b, dim);
    case Metric::Cosine:
      return 1.0F - table.dot(a, b, dim);
  }
  return table.l2sq(a, b, dim);
}

}  // namespace vf::detail
