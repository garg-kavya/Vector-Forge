#pragma once

// Runtime x86 CPU feature detection (CPUID + XGETBV). On other architectures every flag is false.

#include <string>

namespace vf::detail {

struct CpuFeatures {
  bool is_x86 = false;
  std::string vendor;  // e.g. "AuthenticAMD", "GenuineIntel"
  std::string brand;   // processor brand string

  // CPUID-reported instruction sets.
  bool sse2 = false;
  bool sse4_2 = false;
  bool popcnt = false;
  bool avx = false;
  bool fma = false;
  bool avx2 = false;
  bool bmi2 = false;
  bool avx512f = false;

  // Operating-system support for saving extended register state (XCR0).
  bool os_xsave = false;  // CPUID.1:ECX.OSXSAVE
  bool os_ymm = false;    // XCR0 bits 1,2 (SSE + AVX state)
  bool os_zmm = false;    // XCR0 bits 1,2,5,6,7 (AVX-512 state)
};

// Queries the processor. Cheap but not free; prefer cpu_features().
[[nodiscard]] CpuFeatures detect_cpu_features();

// Cached result of detect_cpu_features() (thread-safe initialisation on first call).
[[nodiscard]] const CpuFeatures& cpu_features();

// True when AVX2 and FMA kernels may be executed: CPU support plus OS YMM state saving.
[[nodiscard]] bool can_use_avx2_fma(const CpuFeatures& features) noexcept;

// Multi-line human-readable summary.
[[nodiscard]] std::string describe(const CpuFeatures& features);

}  // namespace vf::detail
