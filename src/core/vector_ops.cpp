#include "core/vector_ops.hpp"

#include <cmath>
#include <cstddef>
#include <span>
#include <string>

#include <vectorforge/distance.hpp>
#include <vectorforge/types.hpp>

#include "core/validation.hpp"
#include "simd/kernels.hpp"
#include "simd/metric_distance.hpp"

namespace vf {

namespace detail {

Result<float> inverse_norm(std::span<const float> vector, const KernelTable& table) {
  if (vector.empty()) {
    return Status::invalid_argument("cannot normalize an empty vector");
  }
  const float norm_sq = table.norm2(vector.data(), vector.size());
  if (!(norm_sq > 0.0F) || !std::isfinite(norm_sq)) {
    return Status::invalid_argument("cannot normalize a vector with zero or non-finite norm");
  }
  // Compute the scale in double precision: the norm of a vector of subnormal components is tiny and
  // its reciprocal can exceed FLT_MAX even though the double value is finite.
  const double inv = 1.0 / std::sqrt(static_cast<double>(norm_sq));
  const auto scale = static_cast<float>(inv);
  if (!std::isfinite(scale)) {
    return Status::invalid_argument("vector norm is too small to normalize");
  }
  return scale;
}

Status normalize_inplace(std::span<float> vector, const KernelTable& table) {
  const Result<float> scale_or = inverse_norm(vector, table);
  if (!scale_or.ok()) {
    return scale_or.status();
  }
  const float scale = scale_or.value();
  for (float& x : vector) {
    x *= scale;
  }
  return {};
}

}  // namespace detail

Result<float> distance(Metric metric, std::span<const float> a, std::span<const float> b) {
  if (!is_valid(metric)) {
    return Status::invalid_argument("invalid metric");
  }
  if (a.size() != b.size()) {
    return Status::dimension_mismatch("vectors have different dimensions (" +
                                      std::to_string(a.size()) + " vs " + std::to_string(b.size()) +
                                      ")");
  }
  if (a.empty()) {
    return Status::invalid_argument("vectors are empty");
  }
  if (detail::find_invalid_component(a) != a.size() ||
      detail::find_invalid_component(b) != b.size()) {
    return Status::invalid_argument("vector contains a NaN, infinite or out-of-range component");
  }

  const detail::KernelTable& table = detail::kernels();
  if (metric != Metric::Cosine) {
    return detail::metric_distance(table, metric, a.data(), b.data(), a.size());
  }

  // Cosine on raw inputs: 1 - dot / (|a| |b|), evaluated in double to avoid overflow of the product
  // of norms.
  const auto norm_a = static_cast<double>(table.norm2(a.data(), a.size()));
  const auto norm_b = static_cast<double>(table.norm2(b.data(), b.size()));
  if (!(norm_a > 0.0) || !(norm_b > 0.0)) {
    return Status::invalid_argument("cosine distance is undefined for a zero vector");
  }
  const auto dot = static_cast<double>(table.dot(a.data(), b.data(), a.size()));
  return static_cast<float>(1.0 - (dot / (std::sqrt(norm_a) * std::sqrt(norm_b))));
}

Status normalize(std::span<float> vector) {
  if (vector.empty()) {
    return Status::invalid_argument("cannot normalize an empty vector");
  }
  const std::size_t bad = detail::find_invalid_component(vector);
  if (bad != vector.size()) {
    return Status::invalid_argument("component " + std::to_string(bad) +
                                    " is NaN, infinite or out of range");
  }
  return detail::normalize_inplace(vector, detail::kernels());
}

}  // namespace vf
