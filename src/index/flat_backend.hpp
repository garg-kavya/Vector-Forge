#pragma once

// Exact (brute-force) k-NN backend (docs/DESIGN.md §9.13).
//
// Scans all rows chunk by chunk in blocks of kBlockRows, computing a block of distances with the
// 1-to-N kernel into a stack buffer and feeding a top-k selector that lives in the caller's output
// span. The query path performs no heap allocation.

#include <cstddef>
#include <span>

#include <vectorforge/types.hpp>

#include "index/index_backend.hpp"
#include "simd/kernels.hpp"
#include "storage/tombstones.hpp"
#include "storage/vector_store.hpp"

namespace vf::detail {

class FlatBackend final : public IndexBackend {
 public:
  // Rows per distance block (stack buffer of kBlockRows floats).
  static constexpr std::size_t kBlockRows = 1024;

  // `vectors` and `deleted` must outlive the backend. `normalized` states that stored rows have
  // unit norm (Cosine, or any metric with normalisation enabled).
  FlatBackend(const VectorStore& vectors, const TombstoneSet& deleted, Metric metric,
              bool normalized, const KernelTable& kernels) noexcept;

  [[nodiscard]] IndexType type() const noexcept override { return IndexType::Flat; }
  [[nodiscard]] Status add(InternalId id) override;
  void remove(InternalId id) noexcept override;
  [[nodiscard]] std::size_t search(const QueryView& query, const SearchKnobs& knobs,
                                   std::span<Neighbor> out) const noexcept override;
  [[nodiscard]] BackendStats stats() const noexcept override { return {}; }

 private:
  // Distances of `query` to `n` contiguous rows, written to `out`.
  void block_distances(const QueryView& query, const float* rows, std::size_t n,
                       float* out) const noexcept;

  const VectorStore* vectors_;
  const TombstoneSet* deleted_;
  Metric metric_;
  bool normalized_;
  const KernelTable* kernels_;
};

}  // namespace vf::detail
