#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <ostream>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

#include <vectorforge/collection.hpp>

#include "commands.hpp"
#include "core/checked_math.hpp"
#include "util/dataset_io.hpp"
#include "util/recall.hpp"
#include "util/timer.hpp"

namespace vf::cli {

namespace {

// Recall against a TEXMEX .ivecs file of true neighbour ids (no distances, so no tie tolerance).
Result<double> ivecs_recall(const std::filesystem::path& path, std::span<const ExternalId> ids,
                            std::uint64_t nq, std::uint32_t k) {
  Result<detail::Matrix<std::int32_t>> gt = detail::read_ivecs(path);
  if (!gt.ok()) {
    return gt.status();
  }
  const detail::Matrix<std::int32_t>& g = gt.value();
  if (g.rows != nq || g.cols < k) {
    return Status::invalid_argument(path.string() + ": expected " + std::to_string(nq) +
                                    " rows with at least k = " + std::to_string(k) + " ids");
  }
  double sum = 0.0;
  std::unordered_set<ExternalId> truth;
  for (std::uint64_t q = 0; q < nq; ++q) {
    truth.clear();
    for (std::uint32_t j = 0; j < k; ++j) {
      truth.insert(static_cast<ExternalId>(g.row(q)[j]));
    }
    std::size_t hits = 0;
    for (std::uint32_t j = 0; j < k; ++j) {
      hits += truth.contains(ids[static_cast<std::size_t>(q * k) + j]) ? 1U : 0U;
    }
    sum += static_cast<double>(hits) / static_cast<double>(k);
  }
  return nq == 0 ? 1.0 : sum / static_cast<double>(nq);
}

Result<detail::RecallStats> prefix_recall(const std::filesystem::path& prefix,
                                          std::span<const ExternalId> ids,
                                          std::span<const float> distances, std::uint64_t nq,
                                          std::uint32_t k) {
  Result<detail::Matrix<std::int64_t>> gt_ids =
      detail::read_npy<std::int64_t>(ground_truth_ids_path(prefix));
  if (!gt_ids.ok()) {
    return gt_ids.status();
  }
  Result<detail::Matrix<float>> gt_dist =
      detail::read_npy<float>(ground_truth_distances_path(prefix));
  if (!gt_dist.ok()) {
    return gt_dist.status();
  }
  const auto& gi = gt_ids.value();
  const auto& gd = gt_dist.value();
  if (gi.rows != nq || gd.rows != nq || gi.cols != gd.cols || gi.cols < k) {
    return Status::invalid_argument("ground truth " + prefix.string() + " has shape " +
                                    std::to_string(gi.rows) + " x " + std::to_string(gi.cols) +
                                    ", expected " + std::to_string(nq) + " rows and >= k columns");
  }
  std::vector<ExternalId> exact(gi.data.size());
  for (std::size_t i = 0; i < exact.size(); ++i) {
    exact[i] = gi.data[i] < 0 ? kInvalidExternalId : static_cast<ExternalId>(gi.data[i]);
  }
  return detail::recall_at_k(ids, distances, k, exact, gd.data, static_cast<std::size_t>(gi.cols),
                             static_cast<std::size_t>(nq), k);
}

}  // namespace

Status run_search(const SearchOptions& options, std::ostream& log) {
  const detail::Stopwatch open_timer;
  Result<std::unique_ptr<Collection>> loaded =
      Collection::load(options.index, {.use_mmap = options.use_mmap});
  if (!loaded.ok()) {
    return loaded.status();
  }
  const double open_seconds = open_timer.elapsed_seconds();
  const Collection& col = *loaded.value();

  Result<detail::Matrix<float>> queries = detail::read_float_matrix(options.queries);
  if (!queries.ok()) {
    return queries.status();
  }
  const detail::Matrix<float>& q = queries.value();
  if (q.cols != col.config().dim) {
    return Status::dimension_mismatch("index has dimension " + std::to_string(col.config().dim) +
                                      " but queries have " + std::to_string(q.cols));
  }
  const auto slots = detail::checked_mul(static_cast<std::size_t>(q.rows), std::size_t{options.k});
  if (!slots) {
    return Status::invalid_argument("queries * k overflows");
  }
  SearchParams params;
  params.k = options.k;
  params.ef_search = options.ef_search;
  VF_RETURN_IF_ERROR(params.validate());

  std::vector<ExternalId> ids(*slots, kInvalidExternalId);
  std::vector<float> distances(*slots, std::numeric_limits<float>::infinity());
  std::vector<Neighbor> hits(options.k);
  std::vector<double> latencies(static_cast<std::size_t>(q.rows));
  double total = 0.0;
  for (std::uint64_t i = 0; i < q.rows; ++i) {
    const detail::Stopwatch sw;
    Result<std::size_t> count = col.search_into(q.row(i), params, hits);
    latencies[static_cast<std::size_t>(i)] = sw.elapsed_seconds();
    if (!count.ok()) {
      return {count.status().code(),
              "query " + std::to_string(i) + ": " + count.status().message()};
    }
    total += latencies[static_cast<std::size_t>(i)];
    for (std::size_t j = 0; j < count.value(); ++j) {
      ids[static_cast<std::size_t>(i * options.k) + j] = hits[j].id;
      distances[static_cast<std::size_t>(i * options.k) + j] = hits[j].distance;
    }
  }
  std::sort(latencies.begin(), latencies.end());
  auto pct = [&latencies](double p) {
    if (latencies.empty()) {
      return 0.0;
    }
    const auto rank =
        static_cast<std::size_t>(p / 100.0 * static_cast<double>(latencies.size() - 1));
    return latencies[rank] * 1e6;
  };

  log << "search: " << q.rows << " queries, k " << options.k;
  if (options.ef_search) {
    log << ", ef_search " << *options.ef_search;
  }
  log << ", " << (options.use_mmap ? "mmap" : "heap") << " load " << open_seconds << " s\n"
      << "  " << (total > 0.0 ? static_cast<double>(q.rows) / total : 0.0)
      << " queries/s (single thread), latency p50 " << pct(50) << " us, p99 " << pct(99) << " us\n";

  if (!options.out_prefix.empty()) {
    std::vector<std::int64_t> ids_i64(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
      ids_i64[i] = ids[i] == kInvalidExternalId ? -1 : static_cast<std::int64_t>(ids[i]);
    }
    VF_RETURN_IF_ERROR(detail::write_npy(ground_truth_ids_path(options.out_prefix),
                                         std::span<const std::int64_t>(ids_i64), q.rows,
                                         options.k));
    VF_RETURN_IF_ERROR(detail::write_npy(ground_truth_distances_path(options.out_prefix),
                                         std::span<const float>(distances), q.rows, options.k));
    log << "  wrote " << ground_truth_ids_path(options.out_prefix).string() << " and "
        << ground_truth_distances_path(options.out_prefix).string() << "\n";
  }
  if (!options.ground_truth.empty()) {
    if (options.ground_truth.extension() == ".ivecs") {
      Result<double> recall = ivecs_recall(options.ground_truth, ids, q.rows, options.k);
      if (!recall.ok()) {
        return recall.status();
      }
      log << "  recall@" << options.k << " " << recall.value() << " (id overlap)\n";
    } else {
      Result<detail::RecallStats> recall =
          prefix_recall(options.ground_truth, ids, distances, q.rows, options.k);
      if (!recall.ok()) {
        return recall.status();
      }
      log << "  recall@" << options.k << " " << recall.value().mean << " (min "
          << recall.value().min << ", tie-tolerant)\n";
    }
  }
  return {};
}

}  // namespace vf::cli
