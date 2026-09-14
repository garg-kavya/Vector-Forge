#include "storage/format.hpp"

#include <algorithm>
#include <cstring>

#include "storage/crc32c.hpp"

namespace vf::detail::format {

namespace {

void put_le(std::span<std::byte> out, std::size_t offset, std::uint64_t value, std::size_t width) {
  for (std::size_t i = 0; i < width; ++i) {
    out[offset + i] = static_cast<std::byte>((value >> (8U * i)) & 0xFFU);
  }
}

std::uint64_t get_le(std::span<const std::byte> in, std::size_t offset,
                     std::size_t width) noexcept {
  std::uint64_t v = 0;
  for (std::size_t i = 0; i < width; ++i) {
    v |= static_cast<std::uint64_t>(in[offset + i]) << (8U * i);
  }
  return v;
}

}  // namespace

std::string_view to_string(SectionType type) noexcept {
  switch (type) {
    case SectionType::Metadata:
      return "METADATA";
    case SectionType::Vectors:
      return "VECTORS";
    case SectionType::Labels:
      return "LABELS";
    case SectionType::Tombstones:
      return "TOMBSTONES";
    case SectionType::Levels:
      return "LEVELS";
    case SectionType::L0Links:
      return "L0_LINKS";
    case SectionType::UpperIndex:
      return "UPPER_INDEX";
    case SectionType::UpperLinks:
      return "UPPER_LINKS";
  }
  return "UNKNOWN";
}

std::array<std::byte, kHeaderSize> encode_header(const FileHeader& h) {
  std::array<std::byte, kHeaderSize> out{};
  std::copy(kMagic.begin(), kMagic.end(), out.begin());
  put_le(out, 8, h.format_major, 2);
  put_le(out, 10, h.format_minor, 2);
  put_le(out, 12, kEndianTag, 4);
  put_le(out, 16, h.flags, 8);
  put_le(out, 24, h.file_size, 8);
  put_le(out, 32, h.section_table_offset, 8);
  put_le(out, 40, h.section_count, 4);
  put_le(out, 44, h.section_table_crc32c, 4);
  // 48: reserved u64 = 0
  const std::uint32_t crc = crc32c(std::span<const std::byte>(out.data(), kHeaderCrcCoverage));
  put_le(out, 56, crc, 4);
  // 60: reserved u32 = 0
  return out;
}

FileHeader decode_header(std::span<const std::byte, kHeaderSize> bytes) noexcept {
  FileHeader h;
  h.format_major = static_cast<std::uint16_t>(get_le(bytes, 8, 2));
  h.format_minor = static_cast<std::uint16_t>(get_le(bytes, 10, 2));
  h.flags = get_le(bytes, 16, 8);
  h.file_size = get_le(bytes, 24, 8);
  h.section_table_offset = get_le(bytes, 32, 8);
  h.section_count = static_cast<std::uint32_t>(get_le(bytes, 40, 4));
  h.section_table_crc32c = static_cast<std::uint32_t>(get_le(bytes, 44, 4));
  h.header_crc32c = static_cast<std::uint32_t>(get_le(bytes, 56, 4));
  return h;
}

bool header_reserved_zero(std::span<const std::byte, kHeaderSize> bytes) noexcept {
  return get_le(bytes, 48, 8) == 0 && get_le(bytes, 60, 4) == 0;
}

void encode_section_entry(BinaryWriter& out, const SectionEntry& e) {
  out.u32(e.type);
  out.u32(e.flags);
  out.u64(e.offset);
  out.u64(e.size);
  out.u32(e.crc32c);
  out.u32(0);
}

bool decode_section_entry(ByteReader& in, SectionEntry& e, bool& reserved_zero) noexcept {
  std::uint32_t reserved = 0;
  const bool ok = in.read_u32(e.type) && in.read_u32(e.flags) && in.read_u64(e.offset) &&
                  in.read_u64(e.size) && in.read_u32(e.crc32c) && in.read_u32(reserved);
  reserved_zero = reserved == 0;
  return ok;
}

std::uint64_t metadata_size(const Metadata& meta) noexcept {
  return 64 + 2 + meta.creator.size() + 8;
}

void encode_metadata(BinaryWriter& out, const Metadata& m) {
  out.u32(m.dim);
  out.u8(m.metric);
  out.u8(m.index_type);
  out.u8(m.normalize);
  out.u8(0);
  out.u64(m.node_count);
  out.u64(m.live_count);
  out.u32(m.m);
  out.u32(m.m0);
  out.u32(m.ef_construction);
  out.u32(m.ef_search);
  out.u8(m.max_level_cap);
  out.u8(m.max_level);
  out.u16(0);
  out.u32(m.entry_point);
  out.u64(m.seed);
  out.u64(0);  // reserved (formerly the level multiplier mL; levels are computed exactly from M)
  out.u16(static_cast<std::uint16_t>(m.creator.size()));
  out.bytes(std::as_bytes(std::span<const char>(m.creator.data(), m.creator.size())));
  out.u64(m.created_unix_ms);
}

Result<Metadata> decode_metadata(std::span<const std::byte> bytes) {
  ByteReader in(bytes);
  Metadata m;
  std::uint8_t pad8 = 0;
  std::uint16_t pad16 = 0;
  std::uint64_t reserved = 0;
  std::uint16_t creator_len = 0;
  const bool fixed_ok = in.read_u32(m.dim) && in.read_u8(m.metric) && in.read_u8(m.index_type) &&
                        in.read_u8(m.normalize) && in.read_u8(pad8) && in.read_u64(m.node_count) &&
                        in.read_u64(m.live_count) && in.read_u32(m.m) && in.read_u32(m.m0) &&
                        in.read_u32(m.ef_construction) && in.read_u32(m.ef_search) &&
                        in.read_u8(m.max_level_cap) && in.read_u8(m.max_level) &&
                        in.read_u16(pad16) && in.read_u32(m.entry_point) && in.read_u64(m.seed) &&
                        in.read_u64(reserved) && in.read_u16(creator_len);
  if (!fixed_ok) {
    return Status::corrupt_data("METADATA: truncated");
  }
  if (pad8 != 0 || pad16 != 0 || reserved != 0) {
    return Status::corrupt_data("METADATA: non-zero padding or reserved field");
  }
  if (creator_len > kMaxCreatorLength) {
    return Status::corrupt_data("METADATA: creator string longer than " +
                                std::to_string(kMaxCreatorLength) + " bytes");
  }
  std::span<const std::byte> creator;
  if (!in.read_view(creator_len, creator) || !in.read_u64(m.created_unix_ms)) {
    return Status::corrupt_data("METADATA: truncated");
  }
  if (in.remaining() != 0) {
    return Status::corrupt_data("METADATA: " + std::to_string(in.remaining()) + " trailing bytes");
  }
  m.creator.resize(creator.size());
  if (!creator.empty()) {
    std::memcpy(m.creator.data(), creator.data(), creator.size());
  }
  return m;
}

}  // namespace vf::detail::format
