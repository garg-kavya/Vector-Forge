#include <cstdint>
#include <limits>
#include <memory>
#include <ostream>
#include <span>
#include <string>
#include <vector>

#include <vectorforge/collection.hpp>

#include "commands.hpp"
#include "core/checked_math.hpp"
#include "util/dataset_io.hpp"
#include "util/timer.hpp"

namespace vf::cli {

std::filesystem::path ground_truth_ids_path(const std::filesystem::path& prefix) {
  std::filesystem::path p = prefix;
  p += ".ids.npy";
  return p;
}

std::filesystem::path ground_truth_distances_path(const std::filesystem::path& prefix) {
  std::filesystem::path p = prefix;
  p += ".distances.npy";
  return p;
}

Status run_ground_truth(const GroundTruthOptions& options, std::ostream& log) {
  if (options.out_prefix.empty()) {
    return Status::invalid_argument("--out is required");
  }
  const detail::Stopwatch load_timer;
  Result<detail::Matrix<float>> base = detail::read_float_matrix(options.base);
  if (!base.ok()) {
    return base.status();
  }
  Result<detail::Matrix<float>> queries = detail::read_float_matrix(options.queries);
  if (!queries.ok()) {
    return queries.status();
  }
  const detail::Matrix<float>& b = base.value();
  const detail::Matrix<float>& q = queries.value();
  if (b.cols != q.cols) {
    return Status::dimension_mismatch("base has dimension " + std::to_string(b.cols) +
                                      " but queries have " + std::to_string(q.cols));
  }
  if (b.cols == 0 || b.cols > kMaxDim) {
    return Status::invalid_argument("unsupported dimension " + std::to_string(b.cols));
  }
  const double load_seconds = load_timer.elapsed_seconds();

  CollectionConfig config;
  config.dim = static_cast<std::uint32_t>(b.cols);
  config.metric = options.metric;
  config.index = IndexType::Flat;
  Result<std::unique_ptr<Collection>> collection = Collection::create(config);
  if (!collection.ok()) {
    return collection.status();
  }
  Collection& col = *collection.value();

  const std::unique_ptr<ThreadPool> pool = make_pool(options.threads);
  const detail::Stopwatch build_timer;
  std::vector<ExternalId> ids(static_cast<std::size_t>(b.rows));
  for (std::size_t i = 0; i < ids.size(); ++i) {
    ids[i] = i;
  }
  const Result<std::size_t> added = col.add_batch(ids, b.data, {}, pool.get());
  if (!added.ok()) {
    return {added.status().code(), "base: " + added.status().message()};
  }
  const double build_seconds = build_timer.elapsed_seconds();

  const auto slots = detail::checked_mul(static_cast<std::size_t>(q.rows), std::size_t{options.k});
  if (!slots) {
    return Status::invalid_argument("queries * k overflows");
  }
  std::vector<ExternalId> out_ids(*slots);
  std::vector<float> out_distances(*slots);
  std::vector<std::uint32_t> counts(static_cast<std::size_t>(q.rows));
  const detail::Stopwatch search_timer;
  SearchParams params;
  params.k = options.k;
  VF_RETURN_IF_ERROR(col.search_batch(q.data, static_cast<std::size_t>(q.rows), params, out_ids,
                                      out_distances, counts, pool.get()));
  const double search_seconds = search_timer.elapsed_seconds();

  std::vector<std::int64_t> ids_i64(out_ids.size());
  for (std::size_t i = 0; i < out_ids.size(); ++i) {
    ids_i64[i] = out_ids[i] == kInvalidExternalId ? -1 : static_cast<std::int64_t>(out_ids[i]);
  }
  VF_RETURN_IF_ERROR(detail::write_npy(ground_truth_ids_path(options.out_prefix),
                                       std::span<const std::int64_t>(ids_i64), q.rows, options.k));
  VF_RETURN_IF_ERROR(detail::write_npy(ground_truth_distances_path(options.out_prefix),
                                       std::span<const float>(out_distances), q.rows, options.k));

  log << "ground truth: base " << b.rows << " x " << b.cols << ", queries " << q.rows << ", k "
      << options.k << ", metric " << to_string(options.metric) << "\n"
      << "  load " << load_seconds << " s, insert " << build_seconds << " s, search "
      << search_seconds << " s (single thread)\n"
      << "  wrote " << ground_truth_ids_path(options.out_prefix).string() << " and "
      << ground_truth_distances_path(options.out_prefix).string() << "\n";
  return {};
}

}  // namespace vf::cli
