#pragma once

// Test-only editor for .vfidx images: locate sections and fields, modify bytes, and recompute every
// checksum so that hostile values reach the reader's semantic validation.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "storage/binary_io.hpp"
#include "storage/crc32c.hpp"
#include "storage/format.hpp"

namespace vf::test {

class VfidxEditor {
 public:
  explicit VfidxEditor(std::vector<std::byte> bytes) : bytes_(std::move(bytes)) { parse(); }

  [[nodiscard]] const std::vector<std::byte>& bytes() const noexcept { return bytes_; }
  [[nodiscard]] std::vector<std::byte>& mutable_bytes() noexcept { return bytes_; }
  [[nodiscard]] const std::vector<detail::format::SectionEntry>& entries() const noexcept {
    return entries_;
  }

  // Index of the table entry for `type`, or -1.
  [[nodiscard]] int entry_index(detail::format::SectionType type) const {
    for (std::size_t i = 0; i < entries_.size(); ++i) {
      if (entries_[i].type == static_cast<std::uint32_t>(type)) {
        return static_cast<int>(i);
      }
    }
    return -1;
  }
  [[nodiscard]] std::uint64_t offset_of(detail::format::SectionType type) const {
    const int i = entry_index(type);
    EXPECT_GE(i, 0);
    return entries_[static_cast<std::size_t>(i)].offset;
  }
  [[nodiscard]] std::uint64_t table_offset() const noexcept { return header_.section_table_offset; }

  void put(std::uint64_t offset, std::uint64_t value, std::size_t width) {
    for (std::size_t i = 0; i < width; ++i) {
      bytes_[static_cast<std::size_t>(offset) + i] =
          static_cast<std::byte>((value >> (8U * i)) & 0xFFU);
    }
  }
  [[nodiscard]] std::uint64_t get(std::uint64_t offset, std::size_t width) const {
    std::uint64_t v = 0;
    for (std::size_t i = 0; i < width; ++i) {
      v |= static_cast<std::uint64_t>(bytes_[static_cast<std::size_t>(offset) + i]) << (8U * i);
    }
    return v;
  }
  // Field of METADATA at a byte offset inside the section (see docs/storage-format.md).
  void put_metadata(std::uint64_t field_offset, std::uint64_t value, std::size_t width) {
    put(offset_of(detail::format::SectionType::Metadata) + field_offset, value, width);
  }
  // Rewrites section table entry `index` from `entry`.
  void put_entry(std::size_t index, const detail::format::SectionEntry& entry) {
    const std::uint64_t base =
        header_.section_table_offset + (index * detail::format::kSectionEntrySize);
    put(base, entry.type, 4);
    put(base + 4, entry.flags, 4);
    put(base + 8, entry.offset, 8);
    put(base + 16, entry.size, 8);
    put(base + 24, entry.crc32c, 4);
    entries_[index] = entry;
  }

  // Recomputes section CRCs (for in-bounds sections), the table CRC and the header CRC.
  void fix_checksums() {
    for (std::size_t i = 0; i < entries_.size(); ++i) {
      detail::format::SectionEntry e = entries_[i];
      if (e.offset <= bytes_.size() && e.size <= bytes_.size() - e.offset) {
        e.crc32c = detail::crc32c(std::span<const std::byte>(bytes_).subspan(
            static_cast<std::size_t>(e.offset), static_cast<std::size_t>(e.size)));
        put_entry(i, e);
      }
    }
    const std::uint64_t table = header_.section_table_offset;
    const std::uint64_t table_size = entries_.size() * detail::format::kSectionEntrySize;
    if (table + table_size <= bytes_.size()) {
      put(44,
          detail::crc32c(std::span<const std::byte>(bytes_).subspan(
              static_cast<std::size_t>(table), static_cast<std::size_t>(table_size))),
          4);
    }
    put(56, detail::crc32c(std::span<const std::byte>(bytes_).first(56)), 4);
  }

 private:
  void parse() {
    ASSERT_GE(bytes_.size(), detail::format::kHeaderSize);
    header_ = detail::format::decode_header(std::span<const std::byte, detail::format::kHeaderSize>(
        bytes_.data(), detail::format::kHeaderSize));
    detail::ByteReader in(std::span<const std::byte>(bytes_).subspan(
        static_cast<std::size_t>(header_.section_table_offset)));
    entries_.resize(header_.section_count);
    for (auto& e : entries_) {
      bool reserved_zero = false;
      ASSERT_TRUE(detail::format::decode_section_entry(in, e, reserved_zero));
    }
  }

  std::vector<std::byte> bytes_;
  detail::format::FileHeader header_;
  std::vector<detail::format::SectionEntry> entries_;
};

}  // namespace vf::test
