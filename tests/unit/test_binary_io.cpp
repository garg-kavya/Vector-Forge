// Binary encoding primitives, fixed format records and the snapshot MANIFEST codec.

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "collection/snapshot.hpp"
#include "storage/binary_io.hpp"
#include "storage/crc32c.hpp"
#include "storage/format.hpp"

namespace {

using vf::ErrorCode;
using vf::detail::BinaryWriter;
using vf::detail::ByteReader;
using vf::detail::MemorySink;
namespace fmt = vf::detail::format;

// Copy of `bytes` with one byte replaced.
std::vector<std::byte> with_byte(std::vector<std::byte> bytes, std::size_t index, unsigned value) {
  bytes.at(index) = static_cast<std::byte>(value);
  return bytes;
}

TEST(BinaryIo, LittleEndianRoundTrip) {
  MemorySink sink;
  BinaryWriter w(sink);
  w.u8(0xAB);
  w.u16(0x1234);
  w.u32(0xDEADBEEF);
  w.u64(0x0102030405060708ULL);
  w.f64(-1.5);
  ASSERT_TRUE(w.status().ok());
  EXPECT_EQ(w.position(), 1U + 2 + 4 + 8 + 8);
  const std::vector<std::byte>& b = sink.bytes();
  EXPECT_EQ(b[1], std::byte{0x34});  // least significant byte first
  EXPECT_EQ(b[2], std::byte{0x12});
  EXPECT_EQ(b[7], std::byte{0x08});

  ByteReader r(b);
  std::uint8_t a = 0;
  std::uint16_t c = 0;
  std::uint32_t d = 0;
  std::uint64_t e = 0;
  double f = 0;
  ASSERT_TRUE(r.read_u8(a) && r.read_u16(c) && r.read_u32(d) && r.read_u64(e) && r.read_f64(f));
  EXPECT_EQ(a, 0xAB);
  EXPECT_EQ(c, 0x1234);
  EXPECT_EQ(d, 0xDEADBEEFU);
  EXPECT_EQ(e, 0x0102030405060708ULL);
  EXPECT_EQ(f, -1.5);
  EXPECT_EQ(r.remaining(), 0U);
}

TEST(BinaryIo, ReaderNeverReadsPastTheEnd) {
  const std::array<std::byte, 5> data{};
  ByteReader r(data);
  std::uint64_t big = 7;
  EXPECT_FALSE(r.read_u64(big));
  EXPECT_EQ(r.position(), 0U) << "a failed read does not advance";
  std::uint32_t mid = 0;
  EXPECT_TRUE(r.read_u32(mid));
  std::uint16_t small = 0;
  EXPECT_FALSE(r.read_u16(small)) << "only one byte left";
  std::span<const std::byte> view;
  EXPECT_FALSE(r.read_view(2, view));
  EXPECT_TRUE(r.read_view(1, view));
  EXPECT_FALSE(r.skip(1));
  EXPECT_TRUE(r.seek(5));
  EXPECT_FALSE(r.seek(6));
  std::array<std::byte, 1> out{};
  EXPECT_FALSE(r.read_bytes(out));
  EXPECT_TRUE(r.seek(0));
  EXPECT_FALSE(r.skip(std::numeric_limits<std::size_t>::max()));
}

TEST(BinaryIo, WriterPaddingCrcAndPatching) {
  MemorySink sink;
  BinaryWriter w(sink);
  w.u8(1);
  w.pad_to(8);
  EXPECT_EQ(w.position(), 8U);
  w.pad_to(8);
  EXPECT_EQ(w.position(), 8U) << "already aligned";
  w.begin_crc();
  const std::string text = "123456789";
  w.bytes(std::as_bytes(std::span<const char>(text.data(), text.size())));
  EXPECT_EQ(w.crc(), 0xE3069283U);
  const std::array<float, 2> floats = {1.0F, -2.0F};
  w.array(std::span<const float>(floats));
  EXPECT_EQ(w.position(), 8U + 9 + 8);
  const std::array<std::byte, 2> patch = {std::byte{9}, std::byte{9}};
  EXPECT_TRUE(sink.write_at(0, patch).ok());
  EXPECT_EQ(sink.bytes()[1], std::byte{9});
  EXPECT_EQ(sink.write_at(24, patch).code(), ErrorCode::IoError) << "beyond written data";
  EXPECT_EQ(fmt::align_up(4097, 4096), 8192U);
  EXPECT_EQ(fmt::align_up(4096, 4096), 4096U);
  EXPECT_FALSE(fmt::align_up(std::numeric_limits<std::uint64_t>::max(), 8).has_value());
}

TEST(Format, HeaderRoundTripAndChecksum) {
  fmt::FileHeader h;
  h.flags = fmt::kFlagHasGraph | fmt::kFlagVectorsNormalized;
  h.file_size = 123456;
  h.section_table_offset = 4096 * 3;
  h.section_count = 7;
  h.section_table_crc32c = 0xCAFEBABE;
  const std::array<std::byte, fmt::kHeaderSize> bytes = fmt::encode_header(h);
  EXPECT_TRUE(std::equal(fmt::kMagic.begin(), fmt::kMagic.end(), bytes.begin()));
  const fmt::FileHeader d = fmt::decode_header(bytes);
  EXPECT_EQ(d.format_major, 1);
  EXPECT_EQ(d.format_minor, 0);
  EXPECT_EQ(d.flags, h.flags);
  EXPECT_EQ(d.file_size, h.file_size);
  EXPECT_EQ(d.section_table_offset, h.section_table_offset);
  EXPECT_EQ(d.section_count, h.section_count);
  EXPECT_EQ(d.section_table_crc32c, h.section_table_crc32c);
  EXPECT_EQ(d.header_crc32c, vf::detail::crc32c(std::span<const std::byte>(bytes).first(56)));
  EXPECT_TRUE(fmt::header_reserved_zero(bytes));
}

TEST(Format, MetadataRoundTripAndStrictness) {
  fmt::Metadata m;
  m.dim = 768;
  m.metric = 2;
  m.index_type = 1;
  m.node_count = 1000;
  m.live_count = 990;
  m.m = 16;
  m.m0 = 32;
  m.ef_construction = 200;
  m.ef_search = 50;
  m.max_level_cap = 16;
  m.max_level = 3;
  m.entry_point = 42;
  m.seed = 99;
  m.creator = "vectorforge 0.1.0";
  m.created_unix_ms = 1700000000000ULL;
  MemorySink sink;
  BinaryWriter w(sink);
  fmt::encode_metadata(w, m);
  ASSERT_EQ(w.position(), fmt::metadata_size(m));
  std::vector<std::byte> bytes = sink.take();
  const vf::Result<fmt::Metadata> d = fmt::decode_metadata(bytes);
  ASSERT_TRUE(d.ok()) << d.status().to_string();
  EXPECT_EQ(d.value().dim, m.dim);
  EXPECT_EQ(d.value().live_count, m.live_count);
  EXPECT_EQ(d.value().entry_point, m.entry_point);
  EXPECT_EQ(d.value().creator, m.creator);
  EXPECT_EQ(d.value().created_unix_ms, m.created_unix_ms);

  EXPECT_EQ(fmt::decode_metadata(std::span<const std::byte>(bytes).first(bytes.size() - 1))
                .status()
                .code(),
            ErrorCode::CorruptData);
  std::vector<std::byte> trailing = bytes;
  trailing.push_back(std::byte{0});
  EXPECT_EQ(fmt::decode_metadata(trailing).status().code(), ErrorCode::CorruptData);
  EXPECT_EQ(fmt::decode_metadata(with_byte(bytes, 7, 1)).status().code(), ErrorCode::CorruptData)
      << "pad byte after normalize";
  EXPECT_EQ(fmt::decode_metadata(with_byte(bytes, 56, 1)).status().code(), ErrorCode::CorruptData)
      << "reserved f64";
  EXPECT_EQ(fmt::decode_metadata(with_byte(with_byte(bytes, 64, 0xFF), 65, 0xFF)).status().code(),
            ErrorCode::CorruptData)
      << "creator length 0xFFFF > 256";
}

TEST(Manifest, EncodeDecode) {
  const vf::detail::Manifest m{
      .format = 1, .generation = 42, .file = "index.000042.vfidx", .crc32c = 3735928559U};
  const std::string text = vf::detail::encode_manifest(m);
  const vf::Result<vf::detail::Manifest> d = vf::detail::decode_manifest(text);
  ASSERT_TRUE(d.ok()) << d.status().to_string();
  EXPECT_EQ(d.value(), m);
  EXPECT_EQ(vf::detail::index_file_name(7), "index.000007.vfidx");
  EXPECT_EQ(vf::detail::index_file_name(12345678), "index.12345678.vfidx");
  // Keys in any order, extra whitespace.
  EXPECT_TRUE(
      vf::detail::decode_manifest(
          " {\"crc32c\":1 ,\n\"file\":\"index.000003.vfidx\",\"generation\":3,\"format\":1}\n")
          .ok());
}

TEST(Manifest, RejectsMalformedInput) {
  const auto code = [](const std::string& text) {
    return vf::detail::decode_manifest(text).status().code();
  };
  EXPECT_EQ(code(""), ErrorCode::CorruptData);
  EXPECT_EQ(code("{}"), ErrorCode::CorruptData);
  EXPECT_EQ(code(R"({"format":1,"generation":3,"file":"index.000003.vfidx"})"),
            ErrorCode::CorruptData);
  EXPECT_EQ(code(R"({"format":2,"generation":3,"file":"index.000003.vfidx","crc32c":1})"),
            ErrorCode::UnsupportedVersion);
  EXPECT_EQ(code(R"({"format":1,"generation":3,"file":"index.000004.vfidx","crc32c":1})"),
            ErrorCode::CorruptData)
      << "file must match generation";
  EXPECT_EQ(code(R"({"format":1,"generation":3,"file":"../index.000003.vfidx","crc32c":1})"),
            ErrorCode::CorruptData)
      << "no path components";
  EXPECT_EQ(code(R"({"format":1,"generation":3,"file":"index.000003.vfidx","crc32c":4294967296})"),
            ErrorCode::CorruptData);
  EXPECT_EQ(code(R"({"format":1,"generation":3,"file":"index.000003.vfidx","crc32c":1,"x":1})"),
            ErrorCode::CorruptData);
  EXPECT_EQ(
      code(R"({"format":1,"format":1,"generation":3,"file":"index.000003.vfidx","crc32c":1})"),
      ErrorCode::CorruptData);
  EXPECT_EQ(code(R"({"format":1,"generation":-3,"file":"index.000003.vfidx","crc32c":1})"),
            ErrorCode::CorruptData);
  EXPECT_EQ(code(R"({"format":1,"generation":3,"file":"index.000003.vfidx","crc32c":1} x)"),
            ErrorCode::CorruptData);
  EXPECT_EQ(code(R"({"format":1,"generation":3,"file":"index.0.vfidx","crc32c":1})"),
            ErrorCode::CorruptData);
}

}  // namespace
