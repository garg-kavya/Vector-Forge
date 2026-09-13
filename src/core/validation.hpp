#pragma once

// Input validation performed once at API boundaries so hot paths can assume clean data.

#include <cstddef>
#include <cstdint>
#include <span>

#include <vectorforge/status.hpp>

namespace vf::detail {

// Index of the first component that is NaN, infinite, or has |x| > kMaxAbsComponent;
// returns values.size() if all components are acceptable.
[[nodiscard]] std::size_t find_invalid_component(std::span<const float> values) noexcept;

// Checks size == expected_dim (DimensionMismatch) and every component (InvalidArgument).
[[nodiscard]] Status validate_vector(std::span<const float> vector, std::uint32_t expected_dim);

// Checks a row-major batch: values.size() == rows * dim and every component valid.
[[nodiscard]] Status validate_batch(std::span<const float> values, std::size_t rows,
                                    std::uint32_t dim);

}  // namespace vf::detail
