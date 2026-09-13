// "Let the compiler vectorise it" kernels, used only as a benchmark baseline against hand-written
// intrinsics. GCC/Clang: `omp simd reduction` (with -fopenmp-simd, no OpenMP runtime).
// MSVC: this translation unit is compiled with /fp:fast, which permits reduction reordering.
// Compiled for the baseline ISA only (SSE2 on x86-64).

#include <cstddef>

#include "simd/kernels.hpp"

namespace vf::detail::scalar_autovec {

float dot(const float* a, const float* b, std::size_t dim) noexcept {
  float sum = 0.0F;
#if defined(VF_USE_OMP_SIMD)
#pragma omp simd reduction(+ : sum)
#endif
  for (std::size_t i = 0; i < dim; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

float l2sq(const float* a, const float* b, std::size_t dim) noexcept {
  float sum = 0.0F;
#if defined(VF_USE_OMP_SIMD)
#pragma omp simd reduction(+ : sum)
#endif
  for (std::size_t i = 0; i < dim; ++i) {
    const float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

float norm2(const float* a, std::size_t dim) noexcept {
  float sum = 0.0F;
#if defined(VF_USE_OMP_SIMD)
#pragma omp simd reduction(+ : sum)
#endif
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

}  // namespace vf::detail::scalar_autovec
