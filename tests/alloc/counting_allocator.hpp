#pragma once

#include <cstddef>

namespace vf::test {

// Counts global operator new calls (all forms) made while an instance is alive.
class AllocationCounter {
 public:
  AllocationCounter() noexcept;
  AllocationCounter(const AllocationCounter&) = delete;
  AllocationCounter& operator=(const AllocationCounter&) = delete;
  AllocationCounter(AllocationCounter&&) = delete;
  AllocationCounter& operator=(AllocationCounter&&) = delete;
  ~AllocationCounter();

  [[nodiscard]] std::size_t count() const noexcept;
};

}  // namespace vf::test
