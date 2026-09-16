#include <string>
#include <string_view>

#include <vectorforge/types.hpp>

namespace vf {

bool is_valid(Metric metric) noexcept {
  switch (metric) {
    case Metric::L2:
    case Metric::InnerProduct:
    case Metric::Cosine:
      return true;
  }
  return false;
}

bool is_valid(IndexType type) noexcept {
  switch (type) {
    case IndexType::Flat:
    case IndexType::Hnsw:
      return true;
  }
  return false;
}

std::string_view to_string(Metric metric) noexcept {
  switch (metric) {
    case Metric::L2:
      return "l2";
    case Metric::InnerProduct:
      return "ip";
    case Metric::Cosine:
      return "cosine";
  }
  return "invalid";
}

std::string_view to_string(IndexType type) noexcept {
  switch (type) {
    case IndexType::Flat:
      return "flat";
    case IndexType::Hnsw:
      return "hnsw";
  }
  return "invalid";
}

std::string_view to_string(Concurrency mode) noexcept {
  switch (mode) {
    case Concurrency::Coarse:
      return "coarse";
    case Concurrency::Concurrent:
      return "concurrent";
  }
  return "invalid";
}

Result<Metric> parse_metric(std::string_view text) {
  if (text == "l2") {
    return Metric::L2;
  }
  if (text == "ip" || text == "inner_product") {
    return Metric::InnerProduct;
  }
  if (text == "cosine") {
    return Metric::Cosine;
  }
  return Status::invalid_argument("unknown metric '" + std::string(text) +
                                  "' (expected l2, ip, inner_product or cosine)");
}

Result<IndexType> parse_index_type(std::string_view text) {
  if (text == "flat") {
    return IndexType::Flat;
  }
  if (text == "hnsw") {
    return IndexType::Hnsw;
  }
  return Status::invalid_argument("unknown index type '" + std::string(text) +
                                  "' (expected flat or hnsw)");
}

Result<Concurrency> parse_concurrency(std::string_view text) {
  if (text == "coarse") {
    return Concurrency::Coarse;
  }
  if (text == "concurrent") {
    return Concurrency::Concurrent;
  }
  return Status::invalid_argument("unknown concurrency mode '" + std::string(text) +
                                  "' (expected coarse or concurrent)");
}

}  // namespace vf
