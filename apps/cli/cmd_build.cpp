#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include <vectorforge/collection.hpp>

#include "commands.hpp"
#include "util/dataset_io.hpp"
#include "util/timer.hpp"

namespace vf::cli {

namespace {

Result<std::vector<ExternalId>> load_ids(const std::filesystem::path& path, std::uint64_t rows) {
  std::vector<ExternalId> ids;
  if (path.empty()) {
    ids.resize(static_cast<std::size_t>(rows));
    for (std::size_t i = 0; i < ids.size(); ++i) {
      ids[i] = i;
    }
    return ids;
  }
  Result<detail::NpyReader> reader = detail::NpyReader::open(path);
  if (!reader.ok()) {
    return reader.status();
  }
  if (reader.value().rows() != rows || reader.value().cols() != 1) {
    return Status::invalid_argument(path.string() + ": expected " + std::to_string(rows) +
                                    " ids in a 1-D array");
  }
  if (reader.value().dtype() == detail::NpyDtype::UInt64) {
    Result<detail::Matrix<std::uint64_t>> m = detail::read_npy<std::uint64_t>(path);
    if (!m.ok()) {
      return m.status();
    }
    return std::move(m.value().data);
  }
  Result<detail::Matrix<std::int64_t>> m = detail::read_npy<std::int64_t>(path);
  if (!m.ok()) {
    return Status::invalid_argument(path.string() + ": ids must be int64 or uint64");
  }
  ids.reserve(m.value().data.size());
  for (const std::int64_t v : m.value().data) {
    if (v < 0) {
      return Status::invalid_argument(path.string() + ": negative id " + std::to_string(v));
    }
    ids.push_back(static_cast<ExternalId>(v));
  }
  return ids;
}

}  // namespace

Status run_build(const BuildOptions& options, std::ostream& log) {
  if (options.out.empty()) {
    return Status::invalid_argument("--out is required");
  }
  const detail::Stopwatch load_timer;
  Result<detail::Matrix<float>> input = detail::read_float_matrix(options.input);
  if (!input.ok()) {
    return input.status();
  }
  const detail::Matrix<float>& data = input.value();
  if (data.cols == 0 || data.cols > kMaxDim) {
    return Status::invalid_argument("unsupported dimension " + std::to_string(data.cols));
  }
  Result<std::vector<ExternalId>> ids = load_ids(options.ids, data.rows);
  if (!ids.ok()) {
    return ids.status();
  }
  const double load_seconds = load_timer.elapsed_seconds();

  CollectionConfig config = options.config;
  config.dim = static_cast<std::uint32_t>(data.cols);
  Result<std::unique_ptr<Collection>> collection = Collection::create(config);
  if (!collection.ok()) {
    return collection.status();
  }
  const std::unique_ptr<ThreadPool> pool = make_pool(options.threads);
  const detail::Stopwatch build_timer;
  const Result<std::size_t> added =
      collection.value()->add_batch(ids.value(), data.data, {}, pool.get());
  if (!added.ok()) {
    return {added.status().code(), options.input.string() + ": " + added.status().message()};
  }
  const double build_seconds = build_timer.elapsed_seconds();
  const detail::Stopwatch save_timer;
  VF_RETURN_IF_ERROR(collection.value()->save(options.out));
  const double save_seconds = save_timer.elapsed_seconds();

  log << "build: " << data.rows << " x " << data.cols << ", metric " << to_string(config.metric)
      << ", index " << to_string(config.index);
  if (config.index == IndexType::Hnsw) {
    log << " (M " << config.hnsw.M << ", ef_construction " << config.hnsw.ef_construction << ")";
  }
  log << "\n  load " << load_seconds << " s, build " << build_seconds << " s (single thread), save "
      << save_seconds << " s\n  wrote " << options.out.string() << "\n";
  return {};
}

}  // namespace vf::cli
