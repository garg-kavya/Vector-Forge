#pragma once

// The .vfidx index file format, major version 1 (docs/storage-format.md, DESIGN.md §12.1).
//
// File = FileHeader (64 bytes) · sections · section table. All integers little-endian.
// This header defines constants and the field-by-field codecs for the fixed records; section
// payloads are produced by the index writer and validated by the index reader.

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <vectorforge/status.hpp>

#include "storage/binary_io.hpp"

namespace vf::detail::format {

inline constexpr std::array<std::byte, 8> kMagic = {
    std::byte{'V'}, std::byte{'F'},  std::byte{'I'},  std::byte{'D'},
    std::byte{'X'}, std::byte{'\r'}, std::byte{'\n'}, std::byte{0x1A}};
inline constexpr std::uint16_t kFormatMajor = 1;
inline constexpr std::uint16_t kFormatMinor = 0;
inline constexpr std::uint32_t kEndianTag = 0x01020304U;
inline constexpr std::uint32_t kEndianTagSwapped = 0x04030201U;

inline constexpr std::size_t kHeaderSize = 64;
inline constexpr std::size_t kHeaderCrcCoverage = 56;  // header CRC covers bytes [0, 56)
inline constexpr std::size_t kSectionEntrySize = 32;
inline constexpr std::uint32_t kMaxSections = 64;
inline constexpr std::uint64_t kSectionAlignment = 8;    // every section offset
inline constexpr std::uint64_t kVectorAlignment = 4096;  // VECTORS offset (page)
inline constexpr std::size_t kMaxCreatorLength = 256;
inline constexpr std::uint32_t kEmptySlot = 0xFFFFFFFFU;  // unused link slot
inline constexpr std::uint64_t kNoUpperBlock = ~std::uint64_t{0};

// FileHeader.flags
inline constexpr std::uint64_t kFlagHasGraph = 1U << 0U;
inline constexpr std::uint64_t kFlagHasTombstones = 1U << 1U;
inline constexpr std::uint64_t kFlagVectorsNormalized = 1U << 2U;
inline constexpr std::uint64_t kKnownFlags =
    kFlagHasGraph | kFlagHasTombstones | kFlagVectorsNormalized;

// SectionEntry.flags
inline constexpr std::uint32_t kSectionOptional = 1U << 0U;  // readers may skip unknown types

// Stored as u32 in the file; the in-memory enum only needs the known values.
enum class SectionType : std::uint8_t {
  Metadata = 1,
  Vectors = 2,
  Labels = 3,
  Tombstones = 4,
  Levels = 5,
  L0Links = 6,
  UpperIndex = 7,
  UpperLinks = 8,
};
inline constexpr std::uint32_t kMaxKnownSectionType = 8;

[[nodiscard]] std::string_view to_string(SectionType type) noexcept;

struct FileHeader {
  std::uint16_t format_major = kFormatMajor;
  std::uint16_t format_minor = kFormatMinor;
  std::uint64_t flags = 0;
  std::uint64_t file_size = 0;
  std::uint64_t section_table_offset = 0;
  std::uint32_t section_count = 0;
  std::uint32_t section_table_crc32c = 0;  // CRC of the section table bytes
  std::uint32_t header_crc32c = 0;         // CRC of header bytes [0, 56)
};

struct SectionEntry {
  std::uint32_t type = 0;
  std::uint32_t flags = 0;
  std::uint64_t offset = 0;
  std::uint64_t size = 0;
  std::uint32_t crc32c = 0;
};

struct Metadata {
  std::uint32_t dim = 0;
  std::uint8_t metric = 0;
  std::uint8_t index_type = 0;
  std::uint8_t normalize = 0;  // CollectionConfig::normalize as configured
  std::uint64_t node_count = 0;
  std::uint64_t live_count = 0;
  // HnswParams (stored for every index type so the configuration round-trips).
  std::uint32_t m = 0;
  std::uint32_t m0 = 0;
  std::uint32_t ef_construction = 0;
  std::uint32_t ef_search = 0;
  std::uint8_t max_level_cap = 0;
  std::uint8_t max_level = 0;  // top level of the graph (entry point level)
  std::uint32_t entry_point = kEmptySlot;
  std::uint64_t seed = 0;
  std::string creator;  // version that created the collection
  std::uint64_t created_unix_ms = 0;
};

// Encodes exactly kHeaderSize bytes, computing header_crc32c from the other fields.
[[nodiscard]] std::array<std::byte, kHeaderSize> encode_header(const FileHeader& header);
// Decodes the fixed fields without validation (the reader validates). Precondition: 64 bytes.
[[nodiscard]] FileHeader decode_header(std::span<const std::byte, kHeaderSize> bytes) noexcept;
// Whether the reserved header fields are zero.
[[nodiscard]] bool header_reserved_zero(std::span<const std::byte, kHeaderSize> bytes) noexcept;

void encode_section_entry(BinaryWriter& out, const SectionEntry& entry);
[[nodiscard]] bool decode_section_entry(ByteReader& in, SectionEntry& entry,
                                        bool& reserved_zero) noexcept;

[[nodiscard]] std::uint64_t metadata_size(const Metadata& meta) noexcept;
void encode_metadata(BinaryWriter& out, const Metadata& meta);
// Errors: CorruptData (truncated, non-zero padding or reserved fields, oversized creator string,
// trailing bytes).
[[nodiscard]] Result<Metadata> decode_metadata(std::span<const std::byte> bytes);

// Round `value` up to a multiple of `alignment` (power of two); nullopt on overflow.
[[nodiscard]] constexpr std::optional<std::uint64_t> align_up(std::uint64_t value,
                                                              std::uint64_t alignment) noexcept {
  const std::uint64_t mask = alignment - 1;
  if (value > ~std::uint64_t{0} - mask) {
    return std::nullopt;
  }
  return (value + mask) & ~mask;
}

}  // namespace vf::detail::format
