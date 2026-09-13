#pragma once

// Monotonic wall-clock stopwatch for tools and benchmarks.

#include <chrono>

namespace vf::detail {

class Stopwatch {
 public:
  using Clock = std::chrono::steady_clock;

  Stopwatch() noexcept : start_(Clock::now()) {}

  void reset() noexcept { start_ = Clock::now(); }

  [[nodiscard]] double elapsed_seconds() const noexcept {
    return std::chrono::duration<double>(Clock::now() - start_).count();
  }

 private:
  Clock::time_point start_;
};

}  // namespace vf::detail
