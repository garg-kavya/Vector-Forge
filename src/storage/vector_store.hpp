#pragma once

// Append-only, chunked float32 row storage (docs/DESIGN.md §4.5, §12.5).
//
// Rows live in fixed-size chunks of `rows_per_chunk` rows (a power of two). A row's address never
// changes once appended, so pointers returned by row_ptr() stay valid for the lifetime of the store
// even as it grows. row(i) = chunk[i >> log2(rows_per_chunk)] + (i & (rows_per_chunk - 1)) * dim.
//
// Heap chunks are 64-byte aligned. A store created by create_mapped() starts with read-only chunks
// that point into caller-owned memory (a memory-mapped index file) kept alive by `owner`; a
// trailing partial chunk is copied to the heap so appends continue in heap chunks after the mapped
// base. Mapped rows are 4-byte aligned (their segment starts on a page boundary).
//
// Thread safety: thread-compatible. Concurrent const access is safe; append()/reserve()/pop_back()
// require exclusive access. Phase 6b replaces the chunk directory with a fixed-capacity atomic
// directory so that readers can run concurrently with appends.
//
// The store does not validate row contents; callers validate at the API boundary.

#include <cstddef>
#include <cstdint>
#include <memory>
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

  // Store whose first `count` rows are `rows[0 .. count * dim)`, which must stay valid and
  // unmodified while `owner` is alive. Full chunks reference that memory; the remaining
  // count % rows_per_chunk rows are copied into a heap chunk.
  // Errors: as create(); ResourceExhausted if count > max_rows. Throws std::bad_alloc.
  [[nodiscard]] static Result<VectorStore> create_mapped(const Options& options,
                                                         std::shared_ptr<const void> owner,
                                                         const float* rows, std::uint64_t count);

  VectorStore(const VectorStore&) = delete;
  VectorStore& operator=(const VectorStore&) = delete;
  VectorStore(VectorStore&& other) noexcept;
  VectorStore& operator=(VectorStore&& other) noexcept;
  ~VectorStore() = default;

  // Appends one row and returns its id (== previous size()).
  // Errors: DimensionMismatch if row.size() != dim(); ResourceExhausted at max_rows().
  // Throws std::bad_alloc if a new chunk cannot be allocated (the store is unchanged).
  [[nodiscard]] Result<InternalId> append(std::span<const float> row);

  // Removes the last row (undoes the most recent append; chunk memory is kept for reuse).
  // Precondition: size() > mapped_rows() and no reader still uses that row. Phase 6b must revisit
  // this for concurrent readers.
  void pop_back() noexcept {
    VF_ASSERT(size_ > mapped_rows_, "VectorStore::pop_back on empty store or mapped row");
    --size_;
  }

  // Ensures capacity() >= rows by allocating chunks up front.
  // Errors: ResourceExhausted if rows > max_rows(). Throws std::bad_alloc.
  [[nodiscard]] Status reserve(std::uint64_t rows);

  // Precondition: id < size().
  [[nodiscard]] const float* row_ptr(InternalId id) const noexcept {
    VF_ASSERT(id < size_, "VectorStore::row_ptr: id out of range");
    return chunk_ptrs_[id >> chunk_shift_] + (static_cast<std::size_t>(id & chunk_mask_) * dim_);
  }
  [[nodiscard]] std::span<const float> row(InternalId id) const noexcept {
    return {row_ptr(id), dim_};
  }

  // Contiguous row data of chunk `chunk` (rows_in_chunk(chunk) * dim() floats). Sequential scans
  // iterate chunks to avoid per-row address computation. Precondition: chunk < chunk_count().
  [[nodiscard]] const float* chunk_data(std::size_t chunk) const noexcept {
    VF_ASSERT(chunk < chunk_ptrs_.size(), "VectorStore::chunk_data: chunk out of range");
    return chunk_ptrs_[chunk];
  }
  // Number of appended rows stored in `chunk` (may be less than rows_per_chunk() for the last one).
  [[nodiscard]] std::size_t rows_in_chunk(std::size_t chunk) const noexcept {
    const std::uint64_t start = static_cast<std::uint64_t>(chunk) * rows_per_chunk_;
    if (start >= size_) {
      return 0;
    }
    const std::uint64_t remaining = size_ - start;
    return remaining < rows_per_chunk_ ? static_cast<std::size_t>(remaining) : rows_per_chunk_;
  }

  [[nodiscard]] std::uint32_t dim() const noexcept { return dim_; }
  [[nodiscard]] std::uint64_t size() const noexcept { return size_; }
  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
  [[nodiscard]] std::uint64_t capacity() const noexcept {
    return static_cast<std::uint64_t>(chunk_ptrs_.size()) * rows_per_chunk_;
  }
  [[nodiscard]] std::uint64_t max_rows() const noexcept { return max_rows_; }
  [[nodiscard]] std::size_t rows_per_chunk() const noexcept { return rows_per_chunk_; }
  [[nodiscard]] std::size_t chunk_count() const noexcept { return chunk_ptrs_.size(); }
  [[nodiscard]] std::size_t chunk_bytes() const noexcept { return chunk_bytes_; }
  // Rows served from the read-only mapped base (a multiple of rows_per_chunk()).
  [[nodiscard]] std::uint64_t mapped_rows() const noexcept { return mapped_rows_; }
  // Bytes held by heap chunks (excludes the small directory and the mapped base).
  [[nodiscard]] std::size_t allocated_bytes() const noexcept {
    return owned_chunks_.size() * chunk_bytes_;
  }
  // Bytes of row data referenced in the mapped base.
  [[nodiscard]] std::size_t mapped_bytes() const noexcept {
    return static_cast<std::size_t>(mapped_rows_) * dim_ * sizeof(float);
  }

 private:
  VectorStore(std::uint32_t dim, std::size_t rows_per_chunk, std::uint64_t max_rows,
              std::size_t chunk_bytes) noexcept;

  void add_chunk();  // throws std::bad_alloc
  [[nodiscard]] float* heap_row_ptr(InternalId id) noexcept;

  std::uint32_t dim_ = 0;
  std::size_t rows_per_chunk_ = 0;
  std::uint32_t chunk_shift_ = 0;
  std::uint64_t chunk_mask_ = 0;
  std::uint64_t max_rows_ = 0;
  std::size_t chunk_bytes_ = 0;
  std::uint64_t size_ = 0;
  std::uint64_t mapped_rows_ = 0;
  std::vector<const float*> chunk_ptrs_;  // mapped chunks first, then heap chunks
  std::vector<AlignedArray<float, kAlignment>> owned_chunks_;
  std::shared_ptr<const void> owner_;  // keeps mapped memory alive
};

}  // namespace vf::detail
