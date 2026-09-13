#pragma once

// Vector operations built on the kernel table (normalisation).

#include <span>

#include <vectorforge/status.hpp>

#include "simd/kernels.hpp"

namespace vf::detail {

// 1 / ||vector|| computed robustly (double precision). Preconditions and errors as for
// normalize_inplace. Allocation-free on success.
[[nodiscard]] Result<float> inverse_norm(std::span<const float> vector, const KernelTable& table);

// Scales `vector` to unit L2 norm using `table.norm2`. Preconditions: components already validated
// (finite, |x| <= kMaxAbsComponent). Errors: InvalidArgument for empty input or a norm that is zero
// or too small to invert without overflow; the vector is unchanged on error.
[[nodiscard]] Status normalize_inplace(std::span<float> vector, const KernelTable& table);

}  // namespace vf::detail
