#pragma once

// Little-endian binary encoding for the index file format (docs/DESIGN.md §12.1).
//
// Values are encoded field by field with explicit byte shifts, never by copying C++ structs, so
// the encoding does not depend on padding, alignment or host layout. Bulk arrays of trivially
// copyable scalars are written as raw bytes, which is only correct on little-endian hosts; this is
// enforced at compile time.
//
// ByteReader checks every read against the buffer end and reports failure instead of reading out
// of bounds. BinaryWriter records the first sink error and turns later writes into no-ops.

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <vectorforge/status.hpp>

#include "storage/crc32c.hpp"

namespace vf::detail {

// NOLINTNEXTLINE(misc-redundant-expression): documents the host requirement
static_assert(std::endian::native == std::endian::little,
              "VectorForge index files are little-endian; big-endian hosts are not supported");

// Bounds-checked sequential/random reader over an immutable byte span.
class ByteReader {
 public:
  explicit ByteReader(std::span<const std::byte> data) noexcept : data_(data) {}

  [[nodiscard]] std::size_t position() const noexcept { return pos_; }
  [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }
  [[nodiscard]] std::size_t remaining() const noexcept { return data_.size() - pos_; }

  // Each read returns false (leaving `out` unspecified and the position unchanged) if fewer
  // bytes remain than required.
  [[nodiscard]] bool read_u8(std::uint8_t& out) noexcept { return read_le(out); }
  [[nodiscard]] bool read_u16(std::uint16_t& out) noexcept { return read_le(out); }
  [[nodiscard]] bool read_u32(std::uint32_t& out) noexcept { return read_le(out); }
  [[nodiscard]] bool read_u64(std::uint64_t& out) noexcept { return read_le(out); }
  [[nodiscard]] bool read_f64(double& out) noexcept {
    std::uint64_t bits = 0;
    if (!read_le(bits)) {
      return false;
    }
    out = std::bit_cast<double>(bits);
    return true;
  }
  [[nodiscard]] bool read_bytes(std::span<std::byte> out) noexcept;
  // View of the next `count` bytes (advances).
  [[nodiscard]] bool read_view(std::size_t count, std::span<const std::byte>& out) noexcept;
  [[nodiscard]] bool skip(std::size_t count) noexcept;
  [[nodiscard]] bool seek(std::size_t position) noexcept;

 private:
  template <class T>
    requires std::is_unsigned_v<T>
  bool read_le(T& out) noexcept {
    if (remaining() < sizeof(T)) {
      return false;
    }
    T value = 0;
    for (std::size_t i = 0; i < sizeof(T); ++i) {
      value = static_cast<T>(value | (static_cast<T>(data_[pos_ + i]) << (8U * i)));
    }
    out = value;
    pos_ += sizeof(T);
    return true;
  }

  std::span<const std::byte> data_;
  std::size_t pos_ = 0;
};

// Destination of encoded bytes.
class ByteSink {
 public:
  ByteSink() = default;
  ByteSink(const ByteSink&) = delete;
  ByteSink& operator=(const ByteSink&) = delete;
  ByteSink(ByteSink&&) = delete;
  ByteSink& operator=(ByteSink&&) = delete;
  virtual ~ByteSink() = default;

  // Appends bytes at the current end. Errors: IoError.
  [[nodiscard]] virtual Status write(std::span<const std::byte> bytes) = 0;
  // Overwrites already written bytes at `offset` (offset + size <= bytes written). Errors: IoError.
  [[nodiscard]] virtual Status write_at(std::uint64_t offset, std::span<const std::byte> bytes) = 0;
};

// In-memory sink (tests, fuzzing, in-memory round trips).
class MemorySink final : public ByteSink {
 public:
  [[nodiscard]] Status write(std::span<const std::byte> bytes) override;
  [[nodiscard]] Status write_at(std::uint64_t offset, std::span<const std::byte> bytes) override;
  [[nodiscard]] const std::vector<std::byte>& bytes() const noexcept { return bytes_; }
  [[nodiscard]] std::vector<std::byte> take() noexcept { return std::move(bytes_); }

 private:
  std::vector<std::byte> bytes_;
};

// Encodes values into a sink, tracking the byte position and an optional running CRC32C.
class BinaryWriter {
 public:
  explicit BinaryWriter(ByteSink& sink) noexcept : sink_(&sink) {}

  void u8(std::uint8_t v) { le(v); }
  void u16(std::uint16_t v) { le(v); }
  void u32(std::uint32_t v) { le(v); }
  void u64(std::uint64_t v) { le(v); }
  void f64(double v) { le(std::bit_cast<std::uint64_t>(v)); }
  void bytes(std::span<const std::byte> data);
  // Raw little-endian bytes of a scalar array.
  template <class T>
    requires std::is_trivially_copyable_v<T> && std::is_arithmetic_v<T>
  void array(std::span<const T> values) {
    bytes(std::as_bytes(values));
  }
  // Zero bytes until position() is a multiple of `alignment` (a power of two).
  void pad_to(std::uint64_t alignment);
  void zeros(std::uint64_t count);

  // Starts accumulating a CRC over subsequently written bytes; crc() returns it.
  void begin_crc() noexcept { crc_ = 0; }
  [[nodiscard]] std::uint32_t crc() const noexcept { return crc_; }

  [[nodiscard]] std::uint64_t position() const noexcept { return position_; }
  // First error reported by the sink, or OK.
  [[nodiscard]] const Status& status() const noexcept { return status_; }
  [[nodiscard]] ByteSink& sink() noexcept { return *sink_; }

 private:
  template <class T>
    requires std::is_unsigned_v<T>
  void le(T v) {
    std::array<std::byte, sizeof(T)> buf{};
    for (std::size_t i = 0; i < sizeof(T); ++i) {
      buf[i] = static_cast<std::byte>((v >> (8U * i)) & 0xFFU);
    }
    bytes(buf);
  }

  ByteSink* sink_;
  Status status_;
  std::uint64_t position_ = 0;
  std::uint32_t crc_ = 0;
};

}  // namespace vf::detail
