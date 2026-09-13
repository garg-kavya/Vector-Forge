#pragma once

// Overflow-checked unsigned arithmetic for sizes derived from untrusted input
// (e.g. rows * dim * sizeof(float)). Returns std::nullopt on overflow.

#include <concepts>
#include <limits>
#include <optional>

namespace vf::detail {

template <std::unsigned_integral T>
[[nodiscard]] constexpr std::optional<T> checked_add(T a, T b) noexcept {
  if (a > std::numeric_limits<T>::max() - b) {
    return std::nullopt;
  }
  return static_cast<T>(a + b);
}

template <std::unsigned_integral T>
[[nodiscard]] constexpr std::optional<T> checked_mul(T a, T b) noexcept {
  if (a != 0 && b > std::numeric_limits<T>::max() / a) {
    return std::nullopt;
  }
  return static_cast<T>(a * b);
}

// a * b * c with overflow detection at each step.
template <std::unsigned_integral T>
[[nodiscard]] constexpr std::optional<T> checked_mul(T a, T b, T c) noexcept {
  const std::optional<T> ab = checked_mul(a, b);
  if (!ab) {
    return std::nullopt;
  }
  return checked_mul(*ab, c);
}

}  // namespace vf::detail
