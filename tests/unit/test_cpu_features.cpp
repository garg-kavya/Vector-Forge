#include <gtest/gtest.h>

#include <iostream>

#include "simd/cpu_features.hpp"

namespace {

using vf::detail::CpuFeatures;

TEST(CpuFeatures, DetectionReportIsConsistent) {
  const CpuFeatures& f = vf::detail::cpu_features();
  // Printed so CI logs and local runs record the detected capabilities.
  std::cout << "[cpu] " << vf::detail::describe(f) << '\n';

#if defined(_M_X64) || defined(__x86_64__)
  EXPECT_TRUE(f.is_x86);
  EXPECT_TRUE(f.sse2) << "SSE2 is part of the x86-64 baseline";
  EXPECT_FALSE(f.vendor.empty());
#endif
  if (f.avx2 && f.os_ymm) {
    EXPECT_TRUE(f.avx) << "AVX2 implies AVX";
  }
  if (f.os_ymm || f.os_zmm) {
    EXPECT_TRUE(f.os_xsave);
  }
  if (f.os_zmm) {
    EXPECT_TRUE(f.os_ymm);
  }
  EXPECT_EQ(vf::detail::can_use_avx2_fma(f), f.avx && f.avx2 && f.fma && f.os_ymm);
}

TEST(CpuFeatures, CachedMatchesFreshDetection) {
  const CpuFeatures fresh = vf::detail::detect_cpu_features();
  const CpuFeatures& cached = vf::detail::cpu_features();
  EXPECT_EQ(fresh.vendor, cached.vendor);
  EXPECT_EQ(fresh.brand, cached.brand);
  EXPECT_EQ(fresh.avx2, cached.avx2);
  EXPECT_EQ(fresh.fma, cached.fma);
  EXPECT_EQ(fresh.avx512f, cached.avx512f);
  EXPECT_EQ(fresh.os_ymm, cached.os_ymm);
}

#if defined(__AVX2__)
TEST(CpuFeatures, CompiledForAvx2ImpliesRuntimeSupport) {
  // If the whole binary was built with AVX2 enabled and is running, the CPU must support it.
  EXPECT_TRUE(vf::detail::cpu_features().avx2);
}
#endif

TEST(CpuFeatures, AvxNotUsableWithoutOsSupport) {
  CpuFeatures f;
  f.is_x86 = true;
  f.avx = f.avx2 = f.fma = true;
  f.os_ymm = false;
  EXPECT_FALSE(vf::detail::can_use_avx2_fma(f));
  f.os_ymm = true;
  EXPECT_TRUE(vf::detail::can_use_avx2_fma(f));
}

}  // namespace
