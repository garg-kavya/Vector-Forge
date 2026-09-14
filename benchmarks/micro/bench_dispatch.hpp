#pragma once

// Interface between the generic dispatch benchmark and its AVX2-compiled half. Plain declarations
// only: bench_dispatch_avx2.cpp is compiled with AVX2 flags and must not share inline code.

#include <cstddef>

namespace vf::bench {

// Sum of squared L2 distances from `query` to rows[ids[i]] for i < count, with the AVX2 acc4
// kernel inlined into the loop (the "per-ISA instantiation of the search loop" alternative).
float inlined_avx2_l2sq_sum(const float* query, const float* const* rows, std::size_t count,
                            std::size_t dim) noexcept;

}  // namespace vf::bench
