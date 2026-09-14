#include "storage/crc32c.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace vf::detail {

namespace {

constexpr std::uint32_t kPolynomial = 0x82F63B78U;  // reflected Castagnoli polynomial

using Tables = std::array<std::array<std::uint32_t, 256>, 8>;

// tables[0] is the classic byte-wise table; tables[k][b] is the CRC of byte b followed by k zero
// bytes, which lets slice-by-8 fold eight input bytes per step.
constexpr Tables make_tables() noexcept {
  Tables t{};
  for (std::uint32_t b = 0; b < 256; ++b) {
    std::uint32_t crc = b;
    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc & 1U) != 0U ? (crc >> 1U) ^ kPolynomial : crc >> 1U;
    }
    t[0][b] = crc;
  }
  for (std::uint32_t b = 0; b < 256; ++b) {
    for (std::size_t k = 1; k < 8; ++k) {
      const std::uint32_t prev = t[k - 1][b];
      t[k][b] = (prev >> 8U) ^ t[0][prev & 0xFFU];
    }
  }
  return t;
}

constexpr Tables kTables = make_tables();

std::uint32_t byte_at(std::span<const std::byte> data, std::size_t i) noexcept {
  return static_cast<std::uint32_t>(data[i]);
}

}  // namespace

std::uint32_t crc32c_extend(std::uint32_t crc, std::span<const std::byte> data) noexcept {
  std::uint32_t c = ~crc;
  std::size_t i = 0;
  const std::size_t n = data.size();
  for (; i + 8 <= n; i += 8) {
    const std::uint32_t lo = c ^ (byte_at(data, i) | (byte_at(data, i + 1) << 8U) |
                                  (byte_at(data, i + 2) << 16U) | (byte_at(data, i + 3) << 24U));
    c = kTables[7][lo & 0xFFU] ^ kTables[6][(lo >> 8U) & 0xFFU] ^ kTables[5][(lo >> 16U) & 0xFFU] ^
        kTables[4][lo >> 24U] ^ kTables[3][byte_at(data, i + 4)] ^
        kTables[2][byte_at(data, i + 5)] ^ kTables[1][byte_at(data, i + 6)] ^
        kTables[0][byte_at(data, i + 7)];
  }
  for (; i < n; ++i) {
    c = (c >> 8U) ^ kTables[0][(c ^ byte_at(data, i)) & 0xFFU];
  }
  return ~c;
}

}  // namespace vf::detail
