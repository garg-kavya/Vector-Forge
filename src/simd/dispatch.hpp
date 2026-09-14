#pragma once

// Kernel tier selection (docs/DESIGN.md §10.4, docs/simd.md).
//
// The process-wide table is resolved once, on first use, from the CPU features and the VF_SIMD
// environment variable:
//   unset, "" or "auto"  best tier this build and machine can run
//   "scalar"             scalar reference kernels
//   "avx2"               AVX2 + FMA kernels; an error if they are not compiled in or not supported
// Requests may only downgrade: an unsupported request is reported, never silently replaced.

#include <span>
#include <string>
#include <string_view>

#include <vectorforge/status.hpp>

#include "simd/cpu_features.hpp"
#include "simd/kernels.hpp"

namespace vf::detail {

// True when kernels_avx2.cpp is part of this build (VF_ENABLE_AVX2 on an x86-64 target).
[[nodiscard]] bool avx2_kernels_compiled() noexcept;

// Table of the "avx2" tier, or nullptr when the kernels are not compiled in or `features` cannot
// run them.
[[nodiscard]] const KernelTable* avx2_kernel_table(const CpuFeatures& features) noexcept;

// Every AVX2 variant (acc4, acc1, acc4_masked) for tests and benchmarks; empty when
// avx2_kernel_table(features) is nullptr.
[[nodiscard]] std::span<const KernelTable> avx2_variant_tables(
    const CpuFeatures& features) noexcept;

// Resolves a VF_SIMD request against `features`.
// Errors: InvalidArgument (unknown request), FailedPrecondition (AVX2 requested but unavailable).
[[nodiscard]] Result<const KernelTable*> select_kernels(std::string_view request,
                                                        const CpuFeatures& features);

struct KernelSelection {
  std::string request;                 // VF_SIMD as read from the environment ("" when unset)
  Status status;                       // error of select_kernels(), ok on success
  const KernelTable* table = nullptr;  // selected table; the scalar table if `status` is an error
};

// Process-wide selection, computed once (thread-safe initialisation).
[[nodiscard]] const KernelSelection& kernel_selection();

}  // namespace vf::detail
