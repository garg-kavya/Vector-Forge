#include "simd/dispatch.hpp"

#include <array>
#include <cstdlib>
#include <span>
#include <string>
#include <string_view>

#include <vectorforge/simd.hpp>
#include <vectorforge/status.hpp>

#include "simd/cpu_features.hpp"
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

Status simd_status() {
  return detail::kernel_selection().status;
}

namespace detail {

namespace {

std::string read_environment(const char* name) {
#if defined(_MSC_VER)
  char* value = nullptr;
  std::size_t length = 0;
  if (_dupenv_s(&value, &length, name) != 0 || value == nullptr) {
    return {};
  }
  std::string out(value);
  std::free(value);  // NOLINT(cppcoreguidelines-no-malloc): _dupenv_s allocates with malloc.
  return out;
#else
  // Read once during single-threaded-safe static initialisation of the selection.
  const char* value = std::getenv(name);  // NOLINT(concurrency-mt-unsafe)
  return value == nullptr ? std::string{} : std::string(value);
#endif
}

#if defined(VF_HAVE_AVX2_KERNELS)
// Default AVX2 variant, chosen from benchmarks/results (docs/simd.md "Choosing the defaults").
constexpr KernelTable kAvx2Table{
    .dot = &avx2::dot_acc4,
    .l2sq = &avx2::l2sq_acc4,
    .norm2 = &avx2::norm2_acc4,
    .dot_1_to_n = &avx2::dot_1_to_n_acc4,
    .l2sq_1_to_n = &avx2::l2sq_1_to_n_acc4,
    .level = SimdLevel::Avx2,
    .name = "avx2",
};

constexpr std::array<KernelTable, 3> kAvx2Variants = {{
    {
        .dot = &avx2::dot_acc4,
        .l2sq = &avx2::l2sq_acc4,
        .norm2 = &avx2::norm2_acc4,
        .dot_1_to_n = &avx2::dot_1_to_n_acc4,
        .l2sq_1_to_n = &avx2::l2sq_1_to_n_acc4,
        .level = SimdLevel::Avx2,
        .name = "avx2_acc4",
    },
    {
        .dot = &avx2::dot_acc1,
        .l2sq = &avx2::l2sq_acc1,
        .norm2 = &avx2::norm2_acc1,
        .dot_1_to_n = &avx2::dot_1_to_n_acc1,
        .l2sq_1_to_n = &avx2::l2sq_1_to_n_acc1,
        .level = SimdLevel::Avx2,
        .name = "avx2_acc1",
    },
    {
        .dot = &avx2::dot_acc4_masked,
        .l2sq = &avx2::l2sq_acc4_masked,
        .norm2 = &avx2::norm2_acc4_masked,
        .dot_1_to_n = &avx2::dot_1_to_n_acc4_masked,
        .l2sq_1_to_n = &avx2::l2sq_1_to_n_acc4_masked,
        .level = SimdLevel::Avx2,
        .name = "avx2_acc4_masked",
    },
}};
#endif

}  // namespace

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

bool avx2_kernels_compiled() noexcept {
#if defined(VF_HAVE_AVX2_KERNELS)
  return true;
#else
  return false;
#endif
}

const KernelTable* avx2_kernel_table(const CpuFeatures& features) noexcept {
#if defined(VF_HAVE_AVX2_KERNELS)
  return can_use_avx2_fma(features) ? &kAvx2Table : nullptr;
#else
  static_cast<void>(features);
  return nullptr;
#endif
}

std::span<const KernelTable> avx2_variant_tables(const CpuFeatures& features) noexcept {
#if defined(VF_HAVE_AVX2_KERNELS)
  if (can_use_avx2_fma(features)) {
    return kAvx2Variants;
  }
#else
  static_cast<void>(features);
#endif
  return {};
}

Result<const KernelTable*> select_kernels(std::string_view request, const CpuFeatures& features) {
  const KernelTable* avx2 = avx2_kernel_table(features);
  if (request.empty() || request == "auto") {
    return avx2 != nullptr ? avx2 : &scalar_kernel_table();
  }
  if (request == "scalar") {
    return &scalar_kernel_table();
  }
  if (request == "avx2") {
    if (avx2 != nullptr) {
      return avx2;
    }
    if (!avx2_kernels_compiled()) {
      return Status::failed_precondition(
          "VF_SIMD=avx2: AVX2 kernels are not compiled into this build (VF_ENABLE_AVX2=OFF or "
          "non-x86-64 target)");
    }
    return Status::failed_precondition(
        "VF_SIMD=avx2: this CPU or operating system does not support AVX2 and FMA");
  }
  return Status::invalid_argument("VF_SIMD='" + std::string(request) +
                                  "' is not one of auto, scalar, avx2");
}

const KernelSelection& kernel_selection() {
  static const KernelSelection selection = [] {
    KernelSelection resolved;
    resolved.request = read_environment("VF_SIMD");
    Result<const KernelTable*> table = select_kernels(resolved.request, cpu_features());
    if (table.ok()) {
      resolved.table = table.value();
    } else {
      resolved.status = table.status();
      resolved.table = &scalar_kernel_table();
    }
    return resolved;
  }();
  return selection;
}

// Allocation failure during the one-time initialisation terminates (noexcept); it can happen at
// most once per process, before any collection exists.
// NOLINTNEXTLINE(bugprone-exception-escape)
const KernelTable& kernels() noexcept {
  return *kernel_selection().table;
}

}  // namespace detail
}  // namespace vf
