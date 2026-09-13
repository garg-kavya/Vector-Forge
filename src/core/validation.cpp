#include "core/validation.hpp"

#include <cmath>
#include <cstddef>
#include <string>

#include <vectorforge/types.hpp>

#include "core/checked_math.hpp"

namespace vf::detail {

namespace {

Status invalid_component_status(std::span<const float> values, std::size_t index,
                                std::size_t row_dim) {
  const float x = values[index];
  std::string where = "component " + std::to_string(index);
  if (row_dim != 0) {
    where =
        "row " + std::to_string(index / row_dim) + " component " + std::to_string(index % row_dim);
  }
  if (std::isnan(x)) {
    return Status::invalid_argument(where + " is NaN");
  }
  if (std::isinf(x)) {
    return Status::invalid_argument(where + " is infinite");
  }
  return Status::invalid_argument(where + " = " + std::to_string(x) +
                                  " exceeds the supported magnitude 1e16");
}

}  // namespace

std::size_t find_invalid_component(std::span<const float> values) noexcept {
  for (std::size_t i = 0; i < values.size(); ++i) {
    // The negated comparison is also true for NaN.
    if (!(std::fabs(values[i]) <= kMaxAbsComponent)) {
      return i;
    }
  }
  return values.size();
}

Status validate_vector(std::span<const float> vector, std::uint32_t expected_dim) {
  if (vector.size() != expected_dim) {
    return Status::dimension_mismatch("expected dimension " + std::to_string(expected_dim) +
                                      ", got " + std::to_string(vector.size()));
  }
  const std::size_t bad = find_invalid_component(vector);
  if (bad != vector.size()) {
    return invalid_component_status(vector, bad, 0);
  }
  return {};
}

Status validate_batch(std::span<const float> values, std::size_t rows, std::uint32_t dim) {
  const auto expected = checked_mul(rows, std::size_t{dim});
  if (!expected || values.size() != *expected) {
    return Status::dimension_mismatch("expected " + std::to_string(rows) + " rows of dimension " +
                                      std::to_string(dim) + ", got " +
                                      std::to_string(values.size()) + " values");
  }
  const std::size_t bad = find_invalid_component(values);
  if (bad != values.size()) {
    return invalid_component_status(values, bad, dim);
  }
  return {};
}

}  // namespace vf::detail
