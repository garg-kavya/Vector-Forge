#pragma once

// Distance kernel interface and dispatch table (docs/DESIGN.md §10).
//
// Kernels are unchecked, noexcept hot-path functions: callers guarantee valid pointers, equal
// dimensions and finite, range-checked input (see core/validation.hpp). Inputs need not be aligned.
//
// ISA isolation rule: translation units compiled with non-baseline instruction-set flags include
// only this header plus standard/intrinsic headers, and export only the extern functions below.

#include <cstddef>
#include <string_view>

#include <vectorforge/simd_level.hpp>

namespace vf::detail {

using PairKernel = float (*)(const float* a, const float* b, std::size_t dim) noexcept;
using NormKernel = float (*)(const float* a, std::size_t dim) noexcept;
// Computes out[i] = kernel(query, rows + i*dim) for i in [0, n).
using BatchKernel = void (*)(const float* query, const float* rows, std::size_t n, std::size_t dim,
                             float* out) noexcept;

struct KernelTable {
  PairKernel dot = nullptr;    // sum(a_i * b_i)
  PairKernel l2sq = nullptr;   // sum((a_i - b_i)^2)
  NormKernel norm2 = nullptr;  // sum(a_i^2)
  BatchKernel dot_1_to_n = nullptr;
  BatchKernel l2sq_1_to_n = nullptr;
  SimdLevel level = SimdLevel::Scalar;
  std::string_view name;
};

// Scalar reference kernels: strict left-to-right summation, no FP contraction. These are the
// numerical oracle for every other tier.
namespace scalar {
float dot(const float* a, const float* b, std::size_t dim) noexcept;
float l2sq(const float* a, const float* b, std::size_t dim) noexcept;
float norm2(const float* a, std::size_t dim) noexcept;
void dot_1_to_n(const float* query, const float* rows, std::size_t n, std::size_t dim,
                float* out) noexcept;
void l2sq_1_to_n(const float* query, const float* rows, std::size_t n, std::size_t dim,
                 float* out) noexcept;
}  // namespace scalar

// Same loops with compiler auto-vectorisation of the reductions permitted (benchmark comparison;
// not selected by dispatch). Results may differ from scalar within floating-point rounding.
namespace scalar_autovec {
float dot(const float* a, const float* b, std::size_t dim) noexcept;
float l2sq(const float* a, const float* b, std::size_t dim) noexcept;
float norm2(const float* a, std::size_t dim) noexcept;
void dot_1_to_n(const float* query, const float* rows, std::size_t n, std::size_t dim,
                float* out) noexcept;
void l2sq_1_to_n(const float* query, const float* rows, std::size_t n, std::size_t dim,
                 float* out) noexcept;
}  // namespace scalar_autovec

[[nodiscard]] const KernelTable& scalar_kernel_table() noexcept;
[[nodiscard]] const KernelTable& scalar_autovec_kernel_table() noexcept;

// Kernel table selected for this process (simd/dispatch.hpp). If the selection failed (invalid or
// unsupported VF_SIMD request) this is the scalar table, and public entry points report the error.
[[nodiscard]] const KernelTable& kernels() noexcept;

// AVX2 + FMA kernels (kernels_avx2.cpp, compiled only when VF_HAVE_AVX2_KERNELS is defined).
// Callers must check that the CPU and OS support AVX2 and FMA before calling any of them. Variants:
// acc4 (four accumulators, scalar tail), acc1 (one accumulator, scalar tail), acc4_masked (four
// accumulators, masked-load tail).
namespace avx2 {
float dot_acc4(const float* a, const float* b, std::size_t dim) noexcept;
float l2sq_acc4(const float* a, const float* b, std::size_t dim) noexcept;
float norm2_acc4(const float* a, std::size_t dim) noexcept;
void dot_1_to_n_acc4(const float* query, const float* rows, std::size_t n, std::size_t dim,
                     float* out) noexcept;
void l2sq_1_to_n_acc4(const float* query, const float* rows, std::size_t n, std::size_t dim,
                      float* out) noexcept;

float dot_acc1(const float* a, const float* b, std::size_t dim) noexcept;
float l2sq_acc1(const float* a, const float* b, std::size_t dim) noexcept;
float norm2_acc1(const float* a, std::size_t dim) noexcept;
void dot_1_to_n_acc1(const float* query, const float* rows, std::size_t n, std::size_t dim,
                     float* out) noexcept;
void l2sq_1_to_n_acc1(const float* query, const float* rows, std::size_t n, std::size_t dim,
                      float* out) noexcept;

float dot_acc4_masked(const float* a, const float* b, std::size_t dim) noexcept;
float l2sq_acc4_masked(const float* a, const float* b, std::size_t dim) noexcept;
float norm2_acc4_masked(const float* a, std::size_t dim) noexcept;
void dot_1_to_n_acc4_masked(const float* query, const float* rows, std::size_t n, std::size_t dim,
                            float* out) noexcept;
void l2sq_1_to_n_acc4_masked(const float* query, const float* rows, std::size_t n, std::size_t dim,
                             float* out) noexcept;
}  // namespace avx2

}  // namespace vf::detail
