#pragma once

// Recall@k against exact ground truth (docs/DESIGN.md §15.2, §16.4).
//
// Tie-tolerant definition: for each query, an approximate hit counts if its id is among the first k
// exact ids, or if its distance is <= the exact k-th distance (+ a relative tolerance of 1e-6), so
// equally distant alternatives are not penalised. Duplicate and padding ids
// (kInvalidExternalId) never count. Per-query recall = hits / min(k, valid exact results); a query
// with no exact results has recall 1.

#include <cstddef>
#include <cstdint>
#include <span>

#include <vectorforge/status.hpp>
#include <vectorforge/types.hpp>

namespace vf::detail {

struct RecallStats {
  double mean = 0.0;
  double min = 0.0;
  std::size_t queries = 0;
};

// approx_* hold nq rows of approx_cols entries; exact_* hold nq rows of exact_cols entries
// (row-major). Errors: InvalidArgument if k == 0, k > approx_cols, k > exact_cols, or spans are too
// small.
[[nodiscard]] Result<RecallStats> recall_at_k(
    std::span<const ExternalId> approx_ids, std::span<const float> approx_distances,
    std::size_t approx_cols, std::span<const ExternalId> exact_ids,
    std::span<const float> exact_distances, std::size_t exact_cols, std::size_t nq, std::size_t k);

}  // namespace vf::detail
