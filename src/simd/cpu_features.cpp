#include "simd/cpu_features.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <string>

#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
#define VF_X86 1
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#include <immintrin.h>
#endif
#endif

namespace vf::detail {

namespace {

#if defined(VF_X86)

struct CpuidRegs {
  std::uint32_t eax = 0;
  std::uint32_t ebx = 0;
  std::uint32_t ecx = 0;
  std::uint32_t edx = 0;
};

CpuidRegs cpuid(std::uint32_t leaf, std::uint32_t subleaf) noexcept {
  CpuidRegs regs;
#if defined(_MSC_VER)
  std::array<int, 4> raw{};
  __cpuidex(raw.data(), static_cast<int>(leaf), static_cast<int>(subleaf));
  regs.eax = static_cast<std::uint32_t>(raw[0]);
  regs.ebx = static_cast<std::uint32_t>(raw[1]);
  regs.ecx = static_cast<std::uint32_t>(raw[2]);
  regs.edx = static_cast<std::uint32_t>(raw[3]);
#else
  unsigned int a = 0;
  unsigned int b = 0;
  unsigned int c = 0;
  unsigned int d = 0;
  __cpuid_count(leaf, subleaf, a, b, c, d);
  regs.eax = a;
  regs.ebx = b;
  regs.ecx = c;
  regs.edx = d;
#endif
  return regs;
}

// XGETBV(0). Precondition: CPUID reports OSXSAVE, otherwise the instruction faults.
#if defined(_MSC_VER)
std::uint64_t read_xcr0() noexcept {
  return _xgetbv(0);
}
#else
__attribute__((target("xsave"))) std::uint64_t read_xcr0() noexcept {
  return static_cast<std::uint64_t>(_xgetbv(0));
}
#endif

constexpr bool bit(std::uint32_t value, unsigned index) noexcept {
  return ((value >> index) & 1U) != 0U;
}

std::string register_chars(std::uint32_t reg) {
  std::array<char, 4> chars{};
  std::memcpy(chars.data(), &reg, sizeof(reg));
  return {chars.data(), chars.size()};
}

#endif  // VF_X86

}  // namespace

CpuFeatures detect_cpu_features() {
  CpuFeatures f;
#if defined(VF_X86)
  f.is_x86 = true;

  const CpuidRegs leaf0 = cpuid(0, 0);
  const std::uint32_t max_leaf = leaf0.eax;
  f.vendor = register_chars(leaf0.ebx) + register_chars(leaf0.edx) + register_chars(leaf0.ecx);

  if (max_leaf >= 1) {
    const CpuidRegs leaf1 = cpuid(1, 0);
    f.sse2 = bit(leaf1.edx, 26);
    f.sse4_2 = bit(leaf1.ecx, 20);
    f.popcnt = bit(leaf1.ecx, 23);
    f.fma = bit(leaf1.ecx, 12);
    f.os_xsave = bit(leaf1.ecx, 27);
    f.avx = bit(leaf1.ecx, 28);
  }
  if (f.os_xsave) {
    const std::uint64_t xcr0 = read_xcr0();
    f.os_ymm = (xcr0 & 0x6U) == 0x6U;
    f.os_zmm = (xcr0 & 0xE6U) == 0xE6U;
  }
  if (max_leaf >= 7) {
    const CpuidRegs leaf7 = cpuid(7, 0);
    f.avx2 = bit(leaf7.ebx, 5);
    f.bmi2 = bit(leaf7.ebx, 8);
    f.avx512f = bit(leaf7.ebx, 16);
  }

  const std::uint32_t max_ext = cpuid(0x80000000U, 0).eax;
  if (max_ext >= 0x80000004U) {
    std::string brand;
    for (std::uint32_t leaf = 0x80000002U; leaf <= 0x80000004U; ++leaf) {
      const CpuidRegs r = cpuid(leaf, 0);
      brand += register_chars(r.eax) + register_chars(r.ebx) + register_chars(r.ecx) +
               register_chars(r.edx);
    }
    const std::size_t nul = brand.find('\0');
    if (nul != std::string::npos) {
      brand.resize(nul);
    }
    const std::size_t first = brand.find_first_not_of(' ');
    const std::size_t last = brand.find_last_not_of(' ');
    f.brand = first == std::string::npos ? std::string{} : brand.substr(first, last - first + 1);
  }
#endif
  return f;
}

const CpuFeatures& cpu_features() {
  static const CpuFeatures features = detect_cpu_features();
  return features;
}

bool can_use_avx2_fma(const CpuFeatures& features) noexcept {
  return features.is_x86 && features.avx && features.avx2 && features.fma && features.os_ymm;
}

std::string describe(const CpuFeatures& f) {
  auto flag = [](bool value) { return value ? "1" : "0"; };
  std::string out;
  out += "arch=";
  out += f.is_x86 ? "x86" : "other";
  out += " vendor='" + f.vendor + "' brand='" + f.brand + "'\n";
  out += std::string("sse2=") + flag(f.sse2) + " sse4_2=" + flag(f.sse4_2) +
         " popcnt=" + flag(f.popcnt) + " avx=" + flag(f.avx) + " fma=" + flag(f.fma) +
         " avx2=" + flag(f.avx2) + " bmi2=" + flag(f.bmi2) + " avx512f=" + flag(f.avx512f) + "\n";
  out += std::string("os_xsave=") + flag(f.os_xsave) + " os_ymm=" + flag(f.os_ymm) +
         " os_zmm=" + flag(f.os_zmm) + " can_use_avx2_fma=" + flag(can_use_avx2_fma(f));
  return out;
}

}  // namespace vf::detail
