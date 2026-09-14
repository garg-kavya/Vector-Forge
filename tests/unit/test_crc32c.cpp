#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "core/rng.hpp"
#include "storage/crc32c.hpp"

namespace {

using vf::detail::crc32c;
using vf::detail::crc32c_extend;

std::span<const std::byte> bytes_of(std::string_view s) {
  return std::as_bytes(std::span<const char>(s.data(), s.size()));
}

// Bit-at-a-time reference implementation of CRC-32C.
std::uint32_t reference_crc(std::span<const std::byte> data) {
  std::uint32_t crc = 0xFFFFFFFFU;
  for (const std::byte b : data) {
    crc ^= static_cast<std::uint32_t>(b);
    for (int i = 0; i < 8; ++i) {
      crc = (crc & 1U) != 0U ? (crc >> 1U) ^ 0x82F63B78U : crc >> 1U;
    }
  }
  return ~crc;
}

TEST(Crc32c, KnownAnswers) {
  EXPECT_EQ(crc32c({}), 0x00000000U);
  EXPECT_EQ(crc32c(bytes_of("123456789")), 0xE3069283U);  // standard check value
  // RFC 3720 appendix B.4 test vectors.
  std::array<std::byte, 32> zeros{};
  EXPECT_EQ(crc32c(zeros), 0x8A9136AAU);
  std::array<std::byte, 32> ones{};
  ones.fill(std::byte{0xFF});
  EXPECT_EQ(crc32c(ones), 0x62A8AB43U);
  std::array<std::byte, 32> ascending{};
  std::array<std::byte, 32> descending{};
  for (std::size_t i = 0; i < 32; ++i) {
    ascending[i] = static_cast<std::byte>(i);
    descending[i] = static_cast<std::byte>(31 - i);
  }
  EXPECT_EQ(crc32c(ascending), 0x46DD794EU);
  EXPECT_EQ(crc32c(descending), 0x113FDB5CU);
}

TEST(Crc32c, MatchesBitwiseReferenceForAllLengthsAndAlignments) {
  vf::detail::Xoshiro256ss rng(11);
  std::vector<std::byte> buffer(300);
  for (std::byte& b : buffer) {
    b = static_cast<std::byte>(rng() & 0xFFU);
  }
  for (std::size_t offset = 0; offset < 8; ++offset) {
    for (std::size_t len = 0; len + offset <= 200; ++len) {
      const auto s = std::span<const std::byte>(buffer).subspan(offset, len);
      ASSERT_EQ(crc32c(s), reference_crc(s)) << "offset " << offset << " len " << len;
    }
  }
}

TEST(Crc32c, ExtendEqualsConcatenation) {
  const std::string_view text = "The quick brown fox jumps over the lazy dog";
  const std::uint32_t whole = crc32c(bytes_of(text));
  for (std::size_t split = 0; split <= text.size(); ++split) {
    const std::uint32_t head = crc32c(bytes_of(text.substr(0, split)));
    EXPECT_EQ(crc32c_extend(head, bytes_of(text.substr(split))), whole) << "split " << split;
  }
}

}  // namespace
