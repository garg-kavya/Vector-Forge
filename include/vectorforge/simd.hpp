#pragma once

// Reports which SIMD kernel tier is in use.

#include <cstdint>
#include <string_view>

namespace vf {

enum class SimdLevel : std::uint8_t {
  Scalar = 0,
  Avx2 = 1,  // AVX2 + FMA (available from Phase 5)
};

// "scalar", "avx2".
[[nodiscard]] std::string_view to_string(SimdLevel level) noexcept;

// Kernel tier selected for this process. Resolved once on first use.
[[nodiscard]] SimdLevel active_simd_level() noexcept;

}  // namespace vf
