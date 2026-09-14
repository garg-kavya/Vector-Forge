// Kernel tier selection (docs/DESIGN.md §10.4-10.5, docs/simd.md).
//
// select_kernels() is tested against synthetic CPU feature sets. The process-wide selection reads
// VF_SIMD once, so its error path is exercised by CTest re-running these tests with VF_SIMD set to
// "scalar", "avx2" (where supported) and an invalid value (tests/CMakeLists.txt).

#include <gtest/gtest.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <vectorforge/collection.hpp>
#include <vectorforge/distance.hpp>
#include <vectorforge/simd.hpp>

#include "simd/cpu_features.hpp"
#include "simd/dispatch.hpp"
#include "simd/kernels.hpp"

namespace {

using vf::ErrorCode;
using vf::SimdLevel;
using vf::detail::CpuFeatures;
using vf::detail::KernelTable;

CpuFeatures with_avx2() {
  CpuFeatures f;
  f.is_x86 = true;
  f.sse2 = true;
  f.avx = true;
  f.avx2 = true;
  f.fma = true;
  f.os_xsave = true;
  f.os_ymm = true;
  return f;
}

CpuFeatures without_avx2() {
  CpuFeatures f = with_avx2();
  f.fma = false;  // AVX2 without FMA (e.g. some VIA/Zhaoxin cores, or a masked hypervisor) is not
                  // enough
  return f;
}

std::string environment_request() {
#if defined(_MSC_VER)
  char* value = nullptr;
  std::size_t length = 0;
  if (_dupenv_s(&value, &length, "VF_SIMD") != 0 || value == nullptr) {
    return {};
  }
  std::string out(value);
  std::free(value);
  return out;
#else
  const char* value = std::getenv("VF_SIMD");
  return value == nullptr ? std::string{} : std::string(value);
#endif
}

TEST(Dispatch, LevelNames) {
  EXPECT_EQ(vf::to_string(SimdLevel::Scalar), "scalar");
  EXPECT_EQ(vf::to_string(SimdLevel::Avx2), "avx2");
}

TEST(Dispatch, AutoPicksBestAvailableTier) {
  for (const char* request : {"", "auto"}) {
    SCOPED_TRACE(request);
    const auto on_avx2 = vf::detail::select_kernels(request, with_avx2());
    ASSERT_TRUE(on_avx2.ok());
    if (vf::detail::avx2_kernels_compiled()) {
      EXPECT_EQ(on_avx2.value()->level, SimdLevel::Avx2);
      EXPECT_EQ(on_avx2.value()->name, "avx2");
    } else {
      EXPECT_EQ(on_avx2.value(), &vf::detail::scalar_kernel_table());
    }
    const auto on_old_cpu = vf::detail::select_kernels(request, without_avx2());
    ASSERT_TRUE(on_old_cpu.ok());
    EXPECT_EQ(on_old_cpu.value(), &vf::detail::scalar_kernel_table());
    const auto non_x86 = vf::detail::select_kernels(request, CpuFeatures{});
    ASSERT_TRUE(non_x86.ok());
    EXPECT_EQ(non_x86.value(), &vf::detail::scalar_kernel_table());
  }
}

TEST(Dispatch, ScalarRequestAlwaysHonoured) {
  for (const CpuFeatures& f : {with_avx2(), without_avx2(), CpuFeatures{}}) {
    const auto table = vf::detail::select_kernels("scalar", f);
    ASSERT_TRUE(table.ok());
    EXPECT_EQ(table.value(), &vf::detail::scalar_kernel_table());
    EXPECT_EQ(table.value()->level, SimdLevel::Scalar);
  }
}

TEST(Dispatch, UnsupportedAvx2RequestIsAnError) {
  for (const CpuFeatures& f : {without_avx2(), CpuFeatures{}}) {
    const auto table = vf::detail::select_kernels("avx2", f);
    ASSERT_FALSE(table.ok());
    EXPECT_EQ(table.status().code(), ErrorCode::FailedPrecondition);
    EXPECT_NE(table.status().message().find("VF_SIMD=avx2"), std::string::npos);
  }
  CpuFeatures no_os_state = with_avx2();
  no_os_state.os_ymm = false;  // CPU support without OS YMM state saving must not be used
  EXPECT_EQ(vf::detail::select_kernels("avx2", no_os_state).status().code(),
            ErrorCode::FailedPrecondition);

  const auto supported = vf::detail::select_kernels("avx2", with_avx2());
  if (vf::detail::avx2_kernels_compiled()) {
    ASSERT_TRUE(supported.ok());
    EXPECT_EQ(supported.value()->level, SimdLevel::Avx2);
  } else {
    EXPECT_EQ(supported.status().code(), ErrorCode::FailedPrecondition);
    EXPECT_NE(supported.status().message().find("not compiled"), std::string::npos);
  }
}

TEST(Dispatch, UnknownRequestIsInvalidArgument) {
  for (const char* request : {"AVX2", "avx512", "sse2", " scalar", "avx2 ", "none"}) {
    const auto table = vf::detail::select_kernels(request, with_avx2());
    ASSERT_FALSE(table.ok()) << request;
    EXPECT_EQ(table.status().code(), ErrorCode::InvalidArgument) << request;
  }
}

TEST(Dispatch, Avx2VariantsAvailableOnlyWithSupport) {
  EXPECT_TRUE(vf::detail::avx2_variant_tables(without_avx2()).empty());
  EXPECT_EQ(vf::detail::avx2_kernel_table(without_avx2()), nullptr);
  const auto variants = vf::detail::avx2_variant_tables(with_avx2());
  if (!vf::detail::avx2_kernels_compiled()) {
    EXPECT_TRUE(variants.empty());
    return;
  }
  ASSERT_EQ(variants.size(), 3U);
  EXPECT_EQ(variants[0].name, "avx2_acc4");
  EXPECT_EQ(variants[1].name, "avx2_acc1");
  EXPECT_EQ(variants[2].name, "avx2_acc4_masked");
  const KernelTable* tier = vf::detail::avx2_kernel_table(with_avx2());
  ASSERT_NE(tier, nullptr);
  bool tier_is_a_variant = false;
  for (const KernelTable& v : variants) {
    EXPECT_EQ(v.level, SimdLevel::Avx2);
    tier_is_a_variant = tier_is_a_variant ||
                        (v.dot == tier->dot && v.l2sq == tier->l2sq && v.norm2 == tier->norm2 &&
                         v.dot_1_to_n == tier->dot_1_to_n && v.l2sq_1_to_n == tier->l2sq_1_to_n);
  }
  EXPECT_TRUE(tier_is_a_variant) << "the avx2 tier must use one benchmarked variant";
}

TEST(Dispatch, ProcessSelectionMatchesEnvironment) {
  const std::string request = environment_request();
  const vf::detail::KernelSelection& selection = vf::detail::kernel_selection();
  std::cout << "[simd] VF_SIMD='" << request
            << "' active=" << vf::to_string(vf::active_simd_level())
            << " status=" << vf::simd_status().to_string() << '\n';
  EXPECT_EQ(selection.request, request);
  EXPECT_EQ(&vf::detail::kernels(), selection.table);

  const auto expected = vf::detail::select_kernels(request, vf::detail::cpu_features());
  if (expected.ok()) {
    EXPECT_TRUE(vf::simd_status().ok());
    EXPECT_EQ(selection.table, expected.value());
  } else {
    EXPECT_EQ(vf::simd_status().code(), expected.status().code());
    EXPECT_EQ(selection.table, &vf::detail::scalar_kernel_table());
  }
  EXPECT_EQ(vf::active_simd_level(), selection.table->level);
  if (request.empty() || request == "auto") {
    const bool avx2 = vf::detail::avx2_kernels_compiled() &&
                      vf::detail::can_use_avx2_fma(vf::detail::cpu_features());
    EXPECT_EQ(vf::active_simd_level(), avx2 ? SimdLevel::Avx2 : SimdLevel::Scalar);
  }
}

TEST(Dispatch, PublicEntryPointsReportSelectionErrors) {
  const vf::Status status = vf::simd_status();
  vf::CollectionConfig config;
  config.dim = 4;
  config.metric = vf::Metric::Cosine;
  const auto created = vf::Collection::create(config);
  std::vector<float> v = {1.0F, 2.0F, 3.0F, 4.0F};
  const auto dist = vf::distance(vf::Metric::L2, v, v);
  const vf::Status normalized = vf::normalize(v);
  if (status.ok()) {
    ASSERT_TRUE(created.ok());
    EXPECT_EQ(created.value()->stats().simd, vf::active_simd_level());
    EXPECT_TRUE(dist.ok());
    EXPECT_TRUE(normalized.ok());
  } else {
    EXPECT_EQ(created.status().code(), status.code());
    EXPECT_EQ(dist.status().code(), status.code());
    EXPECT_EQ(normalized.code(), status.code());
  }
}

}  // namespace
