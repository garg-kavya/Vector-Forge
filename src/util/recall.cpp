#include "util/recall.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "core/checked_math.hpp"

namespace vf::detail {

Result<RecallStats> recall_at_k(std::span<const ExternalId> approx_ids,
                                std::span<const float> approx_distances, std::size_t approx_cols,
                                std::span<const ExternalId> exact_ids,
                                std::span<const float> exact_distances, std::size_t exact_cols,
                                std::size_t nq, std::size_t k) {
  if (k == 0 || k > approx_cols || k > exact_cols) {
    return Status::invalid_argument("recall_at_k: k must be in [1, min(approx_cols, exact_cols)]");
  }
  const auto approx_total = checked_mul(nq, approx_cols);
  const auto exact_total = checked_mul(nq, exact_cols);
  if (!approx_total || !exact_total || approx_ids.size() < *approx_total ||
      approx_distances.size() < *approx_total || exact_ids.size() < *exact_total ||
      exact_distances.size() < *exact_total) {
    return Status::invalid_argument("recall_at_k: input spans are too small");
  }

  RecallStats stats;
  stats.queries = nq;
  stats.min = nq == 0 ? 0.0 : 1.0;
  if (nq == 0) {
    return stats;
  }

  std::vector<ExternalId> truth;
  std::vector<ExternalId> seen;
  truth.reserve(k);
  seen.reserve(k);
  double sum = 0.0;
  for (std::size_t q = 0; q < nq; ++q) {
    truth.clear();
    float kth = -std::numeric_limits<float>::infinity();
    for (std::size_t j = 0; j < k; ++j) {
      const ExternalId id = exact_ids[(q * exact_cols) + j];
      if (id == kInvalidExternalId) {
        break;
      }
      truth.push_back(id);
      kth = exact_distances[(q * exact_cols) + j];
    }
    if (truth.empty()) {
      sum += 1.0;
      continue;
    }
    std::sort(truth.begin(), truth.end());
    const double tolerance = 1e-6 * std::max(1.0, std::fabs(static_cast<double>(kth)));
    const double limit = static_cast<double>(kth) + tolerance;

    seen.clear();
    std::size_t hits = 0;
    for (std::size_t j = 0; j < k; ++j) {
      const ExternalId id = approx_ids[(q * approx_cols) + j];
      if (id == kInvalidExternalId || std::find(seen.begin(), seen.end(), id) != seen.end()) {
        continue;
      }
      seen.push_back(id);
      const bool in_truth = std::binary_search(truth.begin(), truth.end(), id);
      const bool tied = static_cast<double>(approx_distances[(q * approx_cols) + j]) <= limit;
      if (in_truth || tied) {
        ++hits;
      }
    }
    const double recall =
        std::min(1.0, static_cast<double>(hits) / static_cast<double>(truth.size()));
    sum += recall;
    stats.min = std::min(stats.min, recall);
  }
  stats.mean = sum / static_cast<double>(nq);
  return stats;
}

}  // namespace vf::detail
