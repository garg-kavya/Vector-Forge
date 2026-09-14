// AVX2 + FMA distance kernels (docs/DESIGN.md §10.3, docs/simd.md).
//
// This translation unit is the only library code compiled with AVX2/FMA flags
// (cmake/SimdFlags.cmake). ISA isolation rules: it includes only kernels.hpp (declarations) and
// intrinsic/standard headers, keeps every helper in an anonymous namespace, and exports only the
// extern functions declared in vf::detail::avx2. tools/check_isa_leak.py enforces this.
//
// Variants (benchmarked; dispatch picks one as the "avx2" tier):
//   acc4          4 independent accumulators over 32-float blocks, scalar tail
//   acc1          1 accumulator over 8-float blocks, scalar tail
//   acc4_masked   as acc4, but the < 8 float tail is read with _mm256_maskload_ps
//
// Every variant uses a fixed reduction order, so a given variant returns bit-identical results
// for the same inputs regardless of alignment, and the 1-to-N kernels return exactly the per-pair
// results. All kernels are symmetric in their arguments bit for bit (products commute and
// (a - b)^2 == (b - a)^2), which query_distance relies on.

#include <cstddef>

#include "simd/avx2_kernels_inline.hpp"
#include "simd/kernels.hpp"

namespace vf::detail::avx2 {

// acc4 ----------------------------------------------------------------------------------------

float dot_acc4(const float* a, const float* b, std::size_t dim) noexcept {
  return dot_acc4_impl(a, b, dim, Tail::Scalar);
}

float l2sq_acc4(const float* a, const float* b, std::size_t dim) noexcept {
  return l2sq_acc4_impl(a, b, dim, Tail::Scalar);
}

float norm2_acc4(const float* a, std::size_t dim) noexcept {
  return dot_acc4_impl(a, a, dim, Tail::Scalar);
}

void dot_1_to_n_acc4(const float* query, const float* rows, std::size_t n, std::size_t dim,
                     float* out) noexcept {
  for (std::size_t r = 0; r < n; ++r) {
    out[r] = dot_acc4_impl(query, rows + (r * dim), dim, Tail::Scalar);
  }
}

void l2sq_1_to_n_acc4(const float* query, const float* rows, std::size_t n, std::size_t dim,
                      float* out) noexcept {
  for (std::size_t r = 0; r < n; ++r) {
    out[r] = l2sq_acc4_impl(query, rows + (r * dim), dim, Tail::Scalar);
  }
}

// acc1 ----------------------------------------------------------------------------------------

float dot_acc1(const float* a, const float* b, std::size_t dim) noexcept {
  return dot_acc1_impl(a, b, dim);
}

float l2sq_acc1(const float* a, const float* b, std::size_t dim) noexcept {
  return l2sq_acc1_impl(a, b, dim);
}

float norm2_acc1(const float* a, std::size_t dim) noexcept {
  return dot_acc1_impl(a, a, dim);
}

void dot_1_to_n_acc1(const float* query, const float* rows, std::size_t n, std::size_t dim,
                     float* out) noexcept {
  for (std::size_t r = 0; r < n; ++r) {
    out[r] = dot_acc1_impl(query, rows + (r * dim), dim);
  }
}

void l2sq_1_to_n_acc1(const float* query, const float* rows, std::size_t n, std::size_t dim,
                      float* out) noexcept {
  for (std::size_t r = 0; r < n; ++r) {
    out[r] = l2sq_acc1_impl(query, rows + (r * dim), dim);
  }
}

// acc4_masked ---------------------------------------------------------------------------------

float dot_acc4_masked(const float* a, const float* b, std::size_t dim) noexcept {
  return dot_acc4_impl(a, b, dim, Tail::Masked);
}

float l2sq_acc4_masked(const float* a, const float* b, std::size_t dim) noexcept {
  return l2sq_acc4_impl(a, b, dim, Tail::Masked);
}

float norm2_acc4_masked(const float* a, std::size_t dim) noexcept {
  return dot_acc4_impl(a, a, dim, Tail::Masked);
}

void dot_1_to_n_acc4_masked(const float* query, const float* rows, std::size_t n, std::size_t dim,
                            float* out) noexcept {
  for (std::size_t r = 0; r < n; ++r) {
    out[r] = dot_acc4_impl(query, rows + (r * dim), dim, Tail::Masked);
  }
}

void l2sq_1_to_n_acc4_masked(const float* query, const float* rows, std::size_t n, std::size_t dim,
                             float* out) noexcept {
  for (std::size_t r = 0; r < n; ++r) {
    out[r] = l2sq_acc4_impl(query, rows + (r * dim), dim, Tail::Masked);
  }
}

}  // namespace vf::detail::avx2
