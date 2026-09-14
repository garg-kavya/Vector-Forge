#pragma once

// Reports which SIMD kernel tier is in use (docs/simd.md).
//
// The tier is selected once per process, on first use: the best tier the build and CPU support,
// unless the VF_SIMD environment variable requests "scalar" (or "avx2", or "auto"). Requests can
// only downgrade; an invalid or unsupported request makes Collection::create/load, vf::distance
// and vf::normalize fail with simd_status(). All functions are thread-safe.

#include <vectorforge/simd_level.hpp>  // IWYU pragma: export
#include <vectorforge/status.hpp>

namespace vf {

// Kernel tier selected for this process (Scalar while simd_status() is an error).
[[nodiscard]] SimdLevel active_simd_level() noexcept;

// Outcome of the tier selection. Errors: InvalidArgument (VF_SIMD is not auto, scalar or avx2),
// FailedPrecondition (VF_SIMD=avx2 but the kernels are not compiled in or the CPU/OS lacks AVX2 and
// FMA).
[[nodiscard]] Status simd_status();

}  // namespace vf
