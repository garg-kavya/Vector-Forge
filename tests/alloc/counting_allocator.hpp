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

// While alive, the allocation with zero-based index `fail_at` (counting from construction) throws
// std::bad_alloc instead of allocating (nothrow forms return nullptr). Later allocations succeed.
class AllocationFailure {
 public:
  explicit AllocationFailure(std::size_t fail_at) noexcept;
  AllocationFailure(const AllocationFailure&) = delete;
  AllocationFailure& operator=(const AllocationFailure&) = delete;
  AllocationFailure(AllocationFailure&&) = delete;
  AllocationFailure& operator=(AllocationFailure&&) = delete;
  ~AllocationFailure();

  // True if the injected failure happened.
  [[nodiscard]] bool triggered() const noexcept;
};

}  // namespace vf::test
