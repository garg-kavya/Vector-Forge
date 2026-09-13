#include <string_view>

#include <vectorforge/simd.hpp>

#include "simd/kernels.hpp"

namespace vf {

std::string_view to_string(SimdLevel level) noexcept {
  switch (level) {
    case SimdLevel::Scalar:
      return "scalar";
    case SimdLevel::Avx2:
      return "avx2";
  }
  return "invalid";
}

SimdLevel active_simd_level() noexcept {
  return detail::kernels().level;
}

namespace detail {

const KernelTable& scalar_kernel_table() noexcept {
  static constexpr KernelTable kTable{
      .dot = &scalar::dot,
      .l2sq = &scalar::l2sq,
      .norm2 = &scalar::norm2,
      .dot_1_to_n = &scalar::dot_1_to_n,
      .l2sq_1_to_n = &scalar::l2sq_1_to_n,
      .level = SimdLevel::Scalar,
      .name = "scalar",
  };
  return kTable;
}

const KernelTable& scalar_autovec_kernel_table() noexcept {
  static constexpr KernelTable kTable{
      .dot = &scalar_autovec::dot,
      .l2sq = &scalar_autovec::l2sq,
      .norm2 = &scalar_autovec::norm2,
      .dot_1_to_n = &scalar_autovec::dot_1_to_n,
      .l2sq_1_to_n = &scalar_autovec::l2sq_1_to_n,
      .level = SimdLevel::Scalar,
      .name = "scalar_autovec",
  };
  return kTable;
}

const KernelTable& kernels() noexcept {
  // Phase 1: only the scalar tier exists. Phase 5 adds CPU-feature-based selection and the
  // VF_SIMD environment override here.
  return scalar_kernel_table();
}

}  // namespace detail
}  // namespace vf
