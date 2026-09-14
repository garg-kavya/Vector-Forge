#include "storage/binary_io.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <string>

#include "core/checked_math.hpp"

namespace vf::detail {

bool ByteReader::read_bytes(std::span<std::byte> out) noexcept {
  if (remaining() < out.size()) {
    return false;
  }
  if (!out.empty()) {
    std::memcpy(out.data(), data_.data() + pos_, out.size());
  }
  pos_ += out.size();
  return true;
}

bool ByteReader::read_view(std::size_t count, std::span<const std::byte>& out) noexcept {
  if (remaining() < count) {
    return false;
  }
  out = data_.subspan(pos_, count);
  pos_ += count;
  return true;
}

bool ByteReader::skip(std::size_t count) noexcept {
  if (remaining() < count) {
    return false;
  }
  pos_ += count;
  return true;
}

bool ByteReader::seek(std::size_t position) noexcept {
  if (position > data_.size()) {
    return false;
  }
  pos_ = position;
  return true;
}

Status MemorySink::write(std::span<const std::byte> bytes) {
  bytes_.insert(bytes_.end(), bytes.begin(), bytes.end());
  return {};
}

Status MemorySink::write_at(std::uint64_t offset, std::span<const std::byte> bytes) {
  const std::optional<std::uint64_t> end = checked_add(offset, std::uint64_t{bytes.size()});
  if (!end || *end > bytes_.size()) {
    return Status::io_error("MemorySink::write_at beyond written data");
  }
  std::copy(bytes.begin(), bytes.end(), bytes_.begin() + static_cast<std::ptrdiff_t>(offset));
  return {};
}

void BinaryWriter::bytes(std::span<const std::byte> data) {
  if (!status_.ok() || data.empty()) {
    return;
  }
  Status st = sink_->write(data);
  if (!st.ok()) {
    status_ = std::move(st);
    return;
  }
  crc_ = crc32c_extend(crc_, data);
  position_ += data.size();
}

void BinaryWriter::zeros(std::uint64_t count) {
  constexpr std::size_t kBlock = 4096;
  const std::array<std::byte, kBlock> zero{};
  while (count > 0 && status_.ok()) {
    const auto n = static_cast<std::size_t>(std::min<std::uint64_t>(count, kBlock));
    bytes(std::span<const std::byte>(zero.data(), n));
    count -= n;
  }
}

void BinaryWriter::pad_to(std::uint64_t alignment) {
  const std::uint64_t rem = position_ % alignment;
  if (rem != 0) {
    zeros(alignment - rem);
  }
}

}  // namespace vf::detail
