#include <cmath>
#include <string>

#include <vectorforge/config.hpp>

namespace vf {

namespace {

std::string range_message(const char* name, std::uint64_t value, std::uint64_t lo,
                          std::uint64_t hi) {
  return std::string(name) + " = " + std::to_string(value) + " is outside [" + std::to_string(lo) +
         ", " + std::to_string(hi) + "]";
}

}  // namespace

Status HnswParams::validate() const {
  if (M < kMinM || M > kMaxM) {
    return Status::invalid_argument(range_message("hnsw.M", M, kMinM, kMaxM));
  }
  if (ef_construction < M || ef_construction > kMaxEf) {
    return Status::invalid_argument(
        range_message("hnsw.ef_construction", ef_construction, M, kMaxEf) + " (must be >= M)");
  }
  if (ef_search < 1 || ef_search > kMaxEf) {
    return Status::invalid_argument(range_message("hnsw.ef_search", ef_search, 1, kMaxEf));
  }
  if (max_level < 1 || max_level > kMaxLevelCap) {
    return Status::invalid_argument(range_message("hnsw.max_level", max_level, 1, kMaxLevelCap));
  }
  return {};
}

double HnswParams::level_multiplier() const noexcept {
  return 1.0 / std::log(static_cast<double>(M));
}

Status CollectionConfig::validate() const {
  if (dim < 1 || dim > kMaxDim) {
    return Status::invalid_argument(range_message("dim", dim, 1, kMaxDim));
  }
  if (!is_valid(metric)) {
    return Status::invalid_argument("metric has invalid value " +
                                    std::to_string(static_cast<int>(metric)));
  }
  if (!is_valid(index)) {
    return Status::invalid_argument("index type has invalid value " +
                                    std::to_string(static_cast<int>(index)));
  }
  if (index == IndexType::Hnsw) {
    return hnsw.validate();
  }
  return {};
}

}  // namespace vf
