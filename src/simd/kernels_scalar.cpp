// Scalar reference kernels. Compiled with FP contraction disabled (cmake/SimdFlags.cmake) so the
// result is a fixed left-to-right IEEE-754 evaluation on every compiler and architecture.

#include <cstddef>

#include "simd/kernels.hpp"

namespace vf::detail::scalar {

float dot(const float* a, const float* b, std::size_t dim) noexcept {
  float sum = 0.0F;
  for (std::size_t i = 0; i < dim; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

float l2sq(const float* a, const float* b, std::size_t dim) noexcept {
  float sum = 0.0F;
  for (std::size_t i = 0; i < dim; ++i) {
    const float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

float norm2(const float* a, std::size_t dim) noexcept {
  float sum = 0.0F;
  for (std::size_t i = 0; i < dim; ++i) {
    sum += a[i] * a[i];
  }
  return sum;
}

void dot_1_to_n(const float* query, const float* rows, std::size_t n, std::size_t dim,
                float* out) noexcept {
  for (std::size_t r = 0; r < n; ++r) {
    out[r] = dot(query, rows + (r * dim), dim);
  }
}

void l2sq_1_to_n(const float* query, const float* rows, std::size_t n, std::size_t dim,
                 float* out) noexcept {
  for (std::size_t r = 0; r < n; ++r) {
    out[r] = l2sq(query, rows + (r * dim), dim);
  }
}

}  // namespace vf::detail::scalar
