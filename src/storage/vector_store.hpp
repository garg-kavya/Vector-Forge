#pragma once

// Append-only, chunked, cache-line-aligned float32 row storage (docs/DESIGN.md §4.5).
//
// Rows live in fixed-size chunks of `rows_per_chunk` rows (a power of two). A row's address never
// changes once appended, so pointers returned by row_ptr() stay valid for the lifetime of the store
// even as it grows. row(i) = chunk[i >> log2(rows_per_chunk)] + (i & (rows_per_chunk - 1)) * dim.
//
// Thread safety (Phase 1): thread-compatible. Concurrent const access is safe; append()/reserve()
// require exclusive access. Phase 6b replaces the chunk directory with a fixed-capacity atomic
// directory so that readers can run concurrently with appends.
//
// The store does not validate row contents; callers validate at the API boundary.

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <vectorforge/status.hpp>
#include <vectorforge/types.hpp>

#include "core/aligned_alloc.hpp"
#include "core/assert.hpp"

namespace vf::detail {

class VectorStore {
 public:
  static constexpr std::size_t kAlignment = kCacheLineAlignment;
  // Default chunks target roughly this many bytes, bounded by [1, kMaxDefaultRowsPerChunk] rows.
  static constexpr std::size_t kTargetChunkBytes = std::size_t{16} << 20U;  // 16 MiB
  static constexpr std::size_t kMaxDefaultRowsPerChunk = std::size_t{1} << 16U;

  struct Options {
    std::uint32_t dim = 0;
    // 0 selects default_rows_per_chunk(dim). Otherwise must be a power of two.
    std::size_t rows_per_chunk = 0;
    // Upper bound on the number of rows; at most kMaxVectorsPerCollection.
    std::uint64_t max_rows = kMaxVectorsPerCollection;
  };

  // Largest power of two <= kTargetChunkBytes / (dim * 4), clamped to [1, kMaxDefaultRowsPerChunk].
  [[nodiscard]] static std::size_t default_rows_per_chunk(std::uint32_t dim) noexcept;

  [[nodiscard]] static Result<VectorStore> create(const Options& options);

  VectorStore(const VectorStore&) = delete;
  VectorStore& operator=(const VectorStore&) = delete;
  VectorStore(VectorStore&& other) noexcept;
  VectorStore& operator=(VectorStore&& other) noexcept;
  ~VectorStore() = default;

  // Appends one row and returns its id (== previous size()).
  // Errors: DimensionMismatch if row.size() != dim(); ResourceExhausted at max_rows().
  // Throws std::bad_alloc if a new chunk cannot be allocated (the store is unchanged).
  [[nodiscard]] Result<InternalId> append(std::span<const float> row);

  // Ensures capacity() >= rows by allocating chunks up front.
  // Errors: ResourceExhausted if rows > max_rows(). Throws std::bad_alloc.
  [[nodiscard]] Status reserve(std::uint64_t rows);

  // Precondition: id < size().
  [[nodiscard]] const float* row_ptr(InternalId id) const noexcept {
    VF_ASSERT(id < size_, "VectorStore::row_ptr: id out of range");
    return chunks_[id >> chunk_shift_].get() + (static_cast<std::size_t>(id & chunk_mask_) * dim_);
  }
  [[nodiscard]] float* mutable_row_ptr(InternalId id) noexcept {
    VF_ASSERT(id < size_, "VectorStore::mutable_row_ptr: id out of range");
    return chunks_[id >> chunk_shift_].get() + (static_cast<std::size_t>(id & chunk_mask_) * dim_);
  }
  [[nodiscard]] std::span<const float> row(InternalId id) const noexcept {
    return {row_ptr(id), dim_};
  }

  [[nodiscard]] std::uint32_t dim() const noexcept { return dim_; }
  [[nodiscard]] std::uint64_t size() const noexcept { return size_; }
  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
  [[nodiscard]] std::uint64_t capacity() const noexcept {
    return static_cast<std::uint64_t>(chunks_.size()) * rows_per_chunk_;
  }
  [[nodiscard]] std::uint64_t max_rows() const noexcept { return max_rows_; }
  [[nodiscard]] std::size_t rows_per_chunk() const noexcept { return rows_per_chunk_; }
  [[nodiscard]] std::size_t chunk_count() const noexcept { return chunks_.size(); }
  [[nodiscard]] std::size_t chunk_bytes() const noexcept { return chunk_bytes_; }
  // Bytes held by row chunks (excludes the small directory).
  [[nodiscard]] std::size_t allocated_bytes() const noexcept {
    return chunks_.size() * chunk_bytes_;
  }

 private:
  VectorStore(std::uint32_t dim, std::size_t rows_per_chunk, std::uint64_t max_rows,
              std::size_t chunk_bytes) noexcept;

  void add_chunk();  // throws std::bad_alloc

  std::uint32_t dim_ = 0;
  std::size_t rows_per_chunk_ = 0;
  std::uint32_t chunk_shift_ = 0;
  std::uint64_t chunk_mask_ = 0;
  std::uint64_t max_rows_ = 0;
  std::size_t chunk_bytes_ = 0;
  std::uint64_t size_ = 0;
  std::vector<AlignedArray<float, kAlignment>> chunks_;
};

}  // namespace vf::detail
