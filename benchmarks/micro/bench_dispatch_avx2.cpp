// AVX2-compiled half of bench_dispatch.cpp (built only with AVX2 kernels).

#include <cstddef>

#include "bench_dispatch.hpp"
#include "simd/avx2_kernels_inline.hpp"

namespace vf::bench {

float inlined_avx2_l2sq_sum(const float* query, const float* const* rows, std::size_t count,
                            std::size_t dim) noexcept {
  float sum = 0.0F;
  for (std::size_t i = 0; i < count; ++i) {
    sum += detail::avx2::l2sq_acc4_impl(query, rows[i], dim, detail::avx2::Tail::Scalar);
  }
  return sum;
}

}  // namespace vf::bench
