#pragma once

// AVX2 + FMA kernel implementations (templates with internal linkage).
//
// Include ONLY from translation units compiled with AVX2 and FMA enabled: the library's
// kernels_avx2.cpp and the dispatch-level benchmark (benchmarks/micro/bench_dispatch_avx2.cpp).
// Everything here lives in an anonymous namespace so no AVX2-compiled inline function can be merged
// with a baseline copy by the linker (docs/simd.md "ISA isolation").

#if !defined(__AVX2__)
#error "simd/avx2_kernels_inline.hpp requires AVX2 compiler flags"
#endif

#include <cstddef>
#include <immintrin.h>

// Hand-written intrinsics are the purpose of this header.
// NOLINTBEGIN(portability-simd-intrinsics)

namespace vf::detail::avx2 {

namespace {

enum class Op : unsigned char { Dot, L2 };
enum class Tail : unsigned char { Scalar, Masked };

// Sum of the 8 lanes in a fixed order: ((l0+l4) + (l2+l6)) + ((l1+l5) + (l3+l7)).
inline float horizontal_sum(__m256 v) noexcept {
  const __m128 lo = _mm256_castps256_ps128(v);
  const __m128 hi = _mm256_extractf128_ps(v, 1);
  const __m128 pair = _mm_add_ps(lo, hi);
  const __m128 quad = _mm_add_ps(pair, _mm_movehl_ps(pair, pair));
  const __m128 total = _mm_add_ss(quad, _mm_shuffle_ps(quad, quad, 0x55));
  return _mm_cvtss_f32(total);
}

// Mask selecting the first `count` (< 8) lanes.
inline __m256i tail_mask(std::size_t count) noexcept {
  const __m256i lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  return _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(count)), lanes);
}

// acc + x*y (Dot) or acc + (x - y)^2 (L2) over one 8-float block. Norms pass the same pointer
// twice.
template <Op O>
inline __m256 step(__m256 acc, __m256 x, __m256 y) noexcept {
  if constexpr (O == Op::Dot) {
    return _mm256_fmadd_ps(x, y, acc);
  } else {
    const __m256 diff = _mm256_sub_ps(x, y);
    return _mm256_fmadd_ps(diff, diff, acc);
  }
}

template <Op O>
inline __m256 block(__m256 acc, const float* a, const float* b) noexcept {
  return step<O>(acc, _mm256_loadu_ps(a), _mm256_loadu_ps(b));
}

template <Op O>
inline float scalar_tail(const float* a, const float* b, std::size_t i, std::size_t dim,
                         float sum) noexcept {
  for (; i < dim; ++i) {
    const float term = O == Op::Dot ? a[i] * b[i] : (a[i] - b[i]) * (a[i] - b[i]);
    sum += term;
  }
  return sum;
}

// Four accumulators: 32-float blocks feed acc0..acc3; remaining 8-float blocks feed acc0; a masked
// tail feeds acc1. Reduction: horizontal_sum((acc0 + acc1) + (acc2 + acc3)), then the scalar tail.
template <Op O>
inline float acc4(const float* a, const float* b, std::size_t dim, Tail tail) noexcept {
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 32 <= dim; i += 32) {
    acc0 = block<O>(acc0, a + i, b + i);
    acc1 = block<O>(acc1, a + i + 8, b + i + 8);
    acc2 = block<O>(acc2, a + i + 16, b + i + 16);
    acc3 = block<O>(acc3, a + i + 24, b + i + 24);
  }
  for (; i + 8 <= dim; i += 8) {
    acc0 = block<O>(acc0, a + i, b + i);
  }
  if (tail == Tail::Masked && i < dim) {
    // Masked-out lanes are not read (no fault past the end of the row) and load as +0.
    const __m256i mask = tail_mask(dim - i);
    acc1 = step<O>(acc1, _mm256_maskload_ps(a + i, mask), _mm256_maskload_ps(b + i, mask));
    i = dim;
  }
  const float sum =
      horizontal_sum(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
  return scalar_tail<O>(a, b, i, dim, sum);
}

// One accumulator over 8-float blocks.
template <Op O>
inline float acc1(const float* a, const float* b, std::size_t dim) noexcept {
  __m256 acc = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 8 <= dim; i += 8) {
    acc = block<O>(acc, a + i, b + i);
  }
  return scalar_tail<O>(a, b, i, dim, horizontal_sum(acc));
}

inline float dot_acc4_impl(const float* a, const float* b, std::size_t dim, Tail tail) noexcept {
  return acc4<Op::Dot>(a, b, dim, tail);
}
inline float l2sq_acc4_impl(const float* a, const float* b, std::size_t dim, Tail tail) noexcept {
  return acc4<Op::L2>(a, b, dim, tail);
}
inline float dot_acc1_impl(const float* a, const float* b, std::size_t dim) noexcept {
  return acc1<Op::Dot>(a, b, dim);
}
inline float l2sq_acc1_impl(const float* a, const float* b, std::size_t dim) noexcept {
  return acc1<Op::L2>(a, b, dim);
}

}  // namespace

}  // namespace vf::detail::avx2

// NOLINTEND(portability-simd-intrinsics)
