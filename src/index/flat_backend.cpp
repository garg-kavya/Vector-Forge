#include "index/flat_backend.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>

#include "core/assert.hpp"
#include "search/heaps.hpp"

namespace vf::detail {

FlatBackend::FlatBackend(const VectorStore& vectors, const TombstoneSet& deleted, Metric metric,
                         bool normalized, const KernelTable& kernels) noexcept
    : vectors_(&vectors),
      deleted_(&deleted),
      metric_(metric),
      normalized_(normalized),
      kernels_(&kernels) {
}

Status FlatBackend::add(InternalId id) {
  VF_ASSERT(static_cast<std::uint64_t>(id) + 1 == vectors_->size(),
            "FlatBackend::add: row must be the last appended row");
  static_cast<void>(id);
  return {};
}

void FlatBackend::remove(InternalId /*id*/) noexcept {
}

void FlatBackend::block_distances(const QueryView& query, const float* rows, std::size_t n,
                                  float* out) const noexcept {
  const std::size_t dim = vectors_->dim();
  if (!normalized_) {
    if (metric_ == Metric::L2) {
      kernels_->l2sq_1_to_n(query.data, rows, n, dim, out);
      return;
    }
    // InnerProduct (Cosine collections are always normalised).
    kernels_->dot_1_to_n(query.data, rows, n, dim, out);
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = -out[i];
    }
    return;
  }

  // Stored rows x have unit norm; q is the raw query and s = 1/||q||, so with q' = s*q:
  //   Cosine:        1 - <q', x>
  //   L2:            ||q' - x||^2 = 2 - 2<q', x>   (clamped at 0 against rounding)
  //   InnerProduct:  -<q', x>
  // This avoids copying and normalising the query on the hot path.
  kernels_->dot_1_to_n(query.data, rows, n, dim, out);
  const float s = query.inv_norm;
  switch (metric_) {
    case Metric::Cosine:
      for (std::size_t i = 0; i < n; ++i) {
        out[i] = std::max(0.0F, 1.0F - (out[i] * s));
      }
      return;
    case Metric::L2:
      for (std::size_t i = 0; i < n; ++i) {
        out[i] = std::max(0.0F, 2.0F - (2.0F * out[i] * s));
      }
      return;
    case Metric::InnerProduct:
      for (std::size_t i = 0; i < n; ++i) {
        out[i] = -(out[i] * s);
      }
      return;
  }
}

std::size_t FlatBackend::search(const QueryView& query, const SearchKnobs& knobs,
                                std::span<Neighbor> out) const noexcept {
  VF_ASSERT(knobs.k >= 1 && out.size() >= knobs.k, "FlatBackend::search: output too small");
  // Top-k selection uses a bounded max-heap living in the caller's output span. It outperformed a
  // sorted-insertion array for every k measured on scan-like streams
  // (benchmarks/results/2026-09-13_ryzen7-4800h_msvc-release_phase2).
  BoundedMaxHeap<Neighbor> heap(out.first(knobs.k));
  std::array<float, kBlockRows> distances{};
  const bool has_deleted = deleted_->any();
  const std::size_t dim = vectors_->dim();
  std::size_t base_id = 0;
  for (std::size_t chunk = 0; chunk < vectors_->chunk_count(); ++chunk) {
    const std::size_t rows = vectors_->rows_in_chunk(chunk);
    const float* data = vectors_->chunk_data(chunk);
    for (std::size_t start = 0; start < rows; start += kBlockRows) {
      const std::size_t n = std::min(kBlockRows, rows - start);
      block_distances(query, data + (start * dim), n, distances.data());
      for (std::size_t r = 0; r < n; ++r) {
        const auto id = static_cast<InternalId>(base_id + start + r);
        if (has_deleted && deleted_->test(id)) {
          continue;
        }
        const float d = distances[r];
        // Rows are visited in ascending id, so a candidate whose distance equals the current worst
        // retained distance is never better under the (distance, id) order: every retained
        // candidate with that distance has a smaller id. The cheap `d < worst` test is therefore
        // exact.
        if (heap.full() && !(d < heap.top().distance)) {
          continue;
        }
        heap.push(Neighbor{.id = id, .distance = d});
      }
    }
    base_id += rows;
  }
  return heap.sort_ascending();
}

}  // namespace vf::detail
