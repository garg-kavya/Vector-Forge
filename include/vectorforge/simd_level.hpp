#pragma once

// SIMD tier enumeration (dependency-free so that kernel translation units compiled with special
// instruction-set flags can include it; see docs/simd.md "ISA isolation").

#include <cstdint>
#include <string_view>

namespace vf {

enum class SimdLevel : std::uint8_t {
  Scalar = 0,
  Avx2 = 1,  // AVX2 + FMA
};

// "scalar", "avx2".
[[nodiscard]] std::string_view to_string(SimdLevel level) noexcept;

}  // namespace vf
