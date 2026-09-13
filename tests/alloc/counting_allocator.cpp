// Replaces the global allocation functions for the vf_alloc_tests binary only, counting allocations
// while counting is enabled. Used to verify allocation-free hot paths.

#include "counting_allocator.hpp"

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace {

std::atomic<bool> g_counting{false};
std::atomic<std::size_t> g_allocations{0};
std::atomic<bool> g_injecting{false};
std::atomic<std::size_t> g_until_failure{0};
std::atomic<bool> g_failed{false};

// Counts the allocation and throws std::bad_alloc if it is the injected failure.
void note() {
  if (g_counting.load(std::memory_order_relaxed)) {
    g_allocations.fetch_add(1, std::memory_order_relaxed);
  }
  if (g_injecting.load(std::memory_order_relaxed)) {
    if (g_until_failure.load(std::memory_order_relaxed) == 0) {
      g_injecting.store(false);
      g_failed.store(true);
      throw std::bad_alloc();
    }
    g_until_failure.fetch_sub(1, std::memory_order_relaxed);
  }
}

void* allocate(std::size_t size) {
  note();
  void* p = std::malloc(size == 0 ? 1 : size);
  if (p == nullptr) {
    throw std::bad_alloc();
  }
  return p;
}

void* allocate_aligned(std::size_t size, std::align_val_t al) {
  note();
  const auto alignment = static_cast<std::size_t>(al);
  const std::size_t rounded = ((size == 0 ? 1 : size) + alignment - 1) / alignment * alignment;
#if defined(_WIN32)
  void* p = _aligned_malloc(rounded, alignment);
#else
  void* p = std::aligned_alloc(alignment, rounded);
#endif
  if (p == nullptr) {
    throw std::bad_alloc();
  }
  return p;
}

void release_aligned(void* p) noexcept {
#if defined(_WIN32)
  _aligned_free(p);
#else
  std::free(p);
#endif
}

}  // namespace

namespace vf::test {

AllocationCounter::AllocationCounter() noexcept {
  g_allocations.store(0);
  g_counting.store(true);
}

AllocationCounter::~AllocationCounter() {
  g_counting.store(false);
}

std::size_t AllocationCounter::count() const noexcept {
  return g_allocations.load();
}

AllocationFailure::AllocationFailure(std::size_t fail_at) noexcept {
  g_failed.store(false);
  g_until_failure.store(fail_at);
  g_injecting.store(true);
}

AllocationFailure::~AllocationFailure() {
  g_injecting.store(false);
}

bool AllocationFailure::triggered() const noexcept {
  return g_failed.load();
}

}  // namespace vf::test

// NOLINTBEGIN(misc-new-delete-overloads,cert-dcl54-cpp)
void* operator new(std::size_t size) {
  return allocate(size);
}
void* operator new[](std::size_t size) {
  return allocate(size);
}
void* operator new(std::size_t size, const std::nothrow_t& /*tag*/) noexcept {
  try {
    return allocate(size);
  } catch (...) {
    return nullptr;
  }
}
void* operator new[](std::size_t size, const std::nothrow_t& /*tag*/) noexcept {
  try {
    return allocate(size);
  } catch (...) {
    return nullptr;
  }
}
void* operator new(std::size_t size, std::align_val_t al) {
  return allocate_aligned(size, al);
}
void* operator new[](std::size_t size, std::align_val_t al) {
  return allocate_aligned(size, al);
}

void operator delete(void* p) noexcept {
  std::free(p);
}
void operator delete[](void* p) noexcept {
  std::free(p);
}
void operator delete(void* p, std::size_t /*size*/) noexcept {
  std::free(p);
}
void operator delete[](void* p, std::size_t /*size*/) noexcept {
  std::free(p);
}
void operator delete(void* p, const std::nothrow_t& /*tag*/) noexcept {
  std::free(p);
}
void operator delete[](void* p, const std::nothrow_t& /*tag*/) noexcept {
  std::free(p);
}
void operator delete(void* p, std::align_val_t /*al*/) noexcept {
  release_aligned(p);
}
void operator delete[](void* p, std::align_val_t /*al*/) noexcept {
  release_aligned(p);
}
void operator delete(void* p, std::size_t /*size*/, std::align_val_t /*al*/) noexcept {
  release_aligned(p);
}
void operator delete[](void* p, std::size_t /*size*/, std::align_val_t /*al*/) noexcept {
  release_aligned(p);
}
// NOLINTEND(misc-new-delete-overloads,cert-dcl54-cpp)
