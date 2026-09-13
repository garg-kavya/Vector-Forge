#include "storage/vector_store.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include "core/checked_math.hpp"

namespace vf::detail {

std::size_t VectorStore::default_rows_per_chunk(std::uint32_t dim) noexcept {
  if (dim == 0) {
    return 1;
  }
  const std::size_t row_bytes = std::size_t{dim} * sizeof(float);
  const std::size_t rows = std::max<std::size_t>(1, kTargetChunkBytes / row_bytes);
  return std::min(std::bit_floor(rows), kMaxDefaultRowsPerChunk);
}

Result<VectorStore> VectorStore::create(const Options& options) {
  if (options.dim < 1 || options.dim > kMaxDim) {
    return Status::invalid_argument("VectorStore: dim " + std::to_string(options.dim) +
                                    " is outside [1, " + std::to_string(kMaxDim) + "]");
  }
  const std::size_t rows_per_chunk =
      options.rows_per_chunk == 0 ? default_rows_per_chunk(options.dim) : options.rows_per_chunk;
  if (!std::has_single_bit(rows_per_chunk)) {
    return Status::invalid_argument("VectorStore: rows_per_chunk " +
                                    std::to_string(rows_per_chunk) + " is not a power of two");
  }
  if (options.max_rows < 1 || options.max_rows > kMaxVectorsPerCollection) {
    return Status::invalid_argument("VectorStore: max_rows " + std::to_string(options.max_rows) +
                                    " is outside [1, " + std::to_string(kMaxVectorsPerCollection) +
                                    "]");
  }
  const auto chunk_bytes = checked_mul(rows_per_chunk, std::size_t{options.dim}, sizeof(float));
  if (!chunk_bytes) {
    return Status::invalid_argument("VectorStore: chunk size overflows size_t");
  }
  return VectorStore(options.dim, rows_per_chunk, options.max_rows, *chunk_bytes);
}

VectorStore::VectorStore(std::uint32_t dim, std::size_t rows_per_chunk, std::uint64_t max_rows,
                         std::size_t chunk_bytes) noexcept
    : dim_(dim),
      rows_per_chunk_(rows_per_chunk),
      chunk_shift_(static_cast<std::uint32_t>(std::countr_zero(rows_per_chunk))),
      chunk_mask_(static_cast<std::uint64_t>(rows_per_chunk) - 1U),
      max_rows_(max_rows),
      chunk_bytes_(chunk_bytes) {
}

VectorStore::VectorStore(VectorStore&& other) noexcept
    : dim_(other.dim_),
      rows_per_chunk_(other.rows_per_chunk_),
      chunk_shift_(other.chunk_shift_),
      chunk_mask_(other.chunk_mask_),
      max_rows_(other.max_rows_),
      chunk_bytes_(other.chunk_bytes_),
      size_(std::exchange(other.size_, 0)),
      chunks_(std::move(other.chunks_)) {
  other.chunks_.clear();
}

VectorStore& VectorStore::operator=(VectorStore&& other) noexcept {
  if (this != &other) {
    dim_ = other.dim_;
    rows_per_chunk_ = other.rows_per_chunk_;
    chunk_shift_ = other.chunk_shift_;
    chunk_mask_ = other.chunk_mask_;
    max_rows_ = other.max_rows_;
    chunk_bytes_ = other.chunk_bytes_;
    size_ = std::exchange(other.size_, 0);
    chunks_ = std::move(other.chunks_);
    other.chunks_.clear();
  }
  return *this;
}

void VectorStore::add_chunk() {
  // Reserve the directory slot first so that a failure in either allocation leaves no orphan chunk.
  chunks_.reserve(chunks_.size() + 1);
  chunks_.push_back(make_aligned_array<float, kAlignment>(chunk_bytes_ / sizeof(float)));
}

Result<InternalId> VectorStore::append(std::span<const float> row) {
  if (row.size() != dim_) {
    return Status::dimension_mismatch("VectorStore: expected dimension " + std::to_string(dim_) +
                                      ", got " + std::to_string(row.size()));
  }
  if (size_ >= max_rows_) {
    return Status::resource_exhausted("VectorStore: capacity limit of " +
                                      std::to_string(max_rows_) + " rows reached");
  }
  if (size_ == capacity()) {
    add_chunk();
  }
  const auto id = static_cast<InternalId>(size_);
  ++size_;
  std::copy_n(row.data(), dim_, mutable_row_ptr(id));
  return id;
}

Status VectorStore::reserve(std::uint64_t rows) {
  if (rows > max_rows_) {
    return Status::resource_exhausted("VectorStore: cannot reserve " + std::to_string(rows) +
                                      " rows (max_rows " + std::to_string(max_rows_) + ")");
  }
  while (capacity() < rows) {
    add_chunk();
  }
  return {};
}

}  // namespace vf::detail
