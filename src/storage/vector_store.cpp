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

Result<VectorStore> VectorStore::create_mapped(const Options& options,
                                               std::shared_ptr<const void> owner, const float* rows,
                                               std::uint64_t count) {
  Result<VectorStore> created = create(options);
  if (!created.ok()) {
    return created.status();
  }
  VectorStore& store = created.value();
  if (count > store.max_rows_) {
    return Status::resource_exhausted("VectorStore: " + std::to_string(count) +
                                      " mapped rows exceed max_rows " +
                                      std::to_string(store.max_rows_));
  }
  const std::uint64_t full_chunks = count / store.rows_per_chunk_;
  store.chunk_ptrs_.reserve(static_cast<std::size_t>(full_chunks) + 1);
  for (std::uint64_t c = 0; c < full_chunks; ++c) {
    store.chunk_ptrs_.push_back(rows +
                                (static_cast<std::size_t>(c) * store.chunk_bytes_ / sizeof(float)));
  }
  store.mapped_rows_ = full_chunks * store.rows_per_chunk_;
  store.size_ = store.mapped_rows_;
  store.owner_ = std::move(owner);
  const std::uint64_t tail = count - store.mapped_rows_;
  for (std::uint64_t r = 0; r < tail; ++r) {
    const std::size_t offset =
        static_cast<std::size_t>(store.mapped_rows_ + r) * std::size_t{store.dim_};
    Result<InternalId> id = store.append(std::span<const float>(rows + offset, store.dim_));
    if (!id.ok()) {
      return id.status();
    }
  }
  return created;
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
      mapped_rows_(std::exchange(other.mapped_rows_, 0)),
      chunk_ptrs_(std::move(other.chunk_ptrs_)),
      owned_chunks_(std::move(other.owned_chunks_)),
      owner_(std::move(other.owner_)) {
  other.chunk_ptrs_.clear();
  other.owned_chunks_.clear();
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
    mapped_rows_ = std::exchange(other.mapped_rows_, 0);
    chunk_ptrs_ = std::move(other.chunk_ptrs_);
    owned_chunks_ = std::move(other.owned_chunks_);
    owner_ = std::move(other.owner_);
    other.chunk_ptrs_.clear();
    other.owned_chunks_.clear();
  }
  return *this;
}

void VectorStore::add_chunk() {
  // Reserve both directory slots first so that a failure leaves no orphan chunk.
  chunk_ptrs_.reserve(chunk_ptrs_.size() + 1);
  owned_chunks_.reserve(owned_chunks_.size() + 1);
  AlignedArray<float, kAlignment> chunk =
      make_aligned_array<float, kAlignment>(chunk_bytes_ / sizeof(float));
  chunk_ptrs_.push_back(chunk.get());
  owned_chunks_.push_back(std::move(chunk));
}

float* VectorStore::heap_row_ptr(InternalId id) noexcept {
  VF_ASSERT(id >= mapped_rows_ && id < capacity(), "VectorStore: row is not in a heap chunk");
  // Heap chunks follow the mapped chunks in the directory.
  const auto heap_chunk =
      static_cast<std::size_t>((id >> chunk_shift_) - (mapped_rows_ >> chunk_shift_));
  return owned_chunks_[heap_chunk].get() + (static_cast<std::size_t>(id & chunk_mask_) * dim_);
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
  std::copy_n(row.data(), dim_, heap_row_ptr(id));
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
