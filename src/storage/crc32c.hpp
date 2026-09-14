#pragma once

// CRC-32C (Castagnoli, reflected polynomial 0x82F63B78) as used by iSCSI (RFC 3720), ext4 and
// LevelDB. Portable table-driven slice-by-8 implementation (docs/DESIGN.md §12.1).
//
// crc32c_extend(crc, data) continues a CRC: crc32c(a + b) == crc32c_extend(crc32c(a), b), and
// crc32c_extend(0, data) == crc32c(data). Detects accidental corruption only; not a MAC.

#include <cstddef>
#include <cstdint>
#include <span>

namespace vf::detail {

[[nodiscard]] std::uint32_t crc32c_extend(std::uint32_t crc,
                                          std::span<const std::byte> data) noexcept;

[[nodiscard]] inline std::uint32_t crc32c(std::span<const std::byte> data) noexcept {
  return crc32c_extend(0, data);
}

}  // namespace vf::detail
