#pragma once

// Convenience distance functions for library users. They validate their inputs and use the
// active SIMD kernel tier. Index internals use the unchecked kernel table directly.

#include <span>

#include <vectorforge/status.hpp>
#include <vectorforge/types.hpp>

namespace vf {

// Lower-is-better distance between two vectors under `metric` (see Metric for definitions).
// Cosine is computed from the raw inputs (they need not be normalised).
// Errors: DimensionMismatch if sizes differ; InvalidArgument for empty input, non-finite or
// out-of-range components (|x| > kMaxAbsComponent), or a zero vector under Cosine.
[[nodiscard]] Result<float> distance(Metric metric, std::span<const float> a,
                                     std::span<const float> b);

// Scales `vector` to unit Euclidean length in place.
// Errors: InvalidArgument for empty input, non-finite/out-of-range components, or a norm too small
// to normalise without overflow. On error the vector is left unchanged.
[[nodiscard]] Status normalize(std::span<float> vector);

}  // namespace vf
