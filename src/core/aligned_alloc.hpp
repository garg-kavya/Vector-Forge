#pragma once

// Over-aligned heap allocation for trivially copyable element types (vector storage, scratch
// buffers). Uses C++17 aligned operator new, which is portable across MSVC, libstdc++ (including
// MinGW) and libc++, and understood by AddressSanitizer.
//
// Allocation failure throws std::bad_alloc; element-count overflow throws
// std::bad_array_new_length. Both are treated as fatal by the frontends (docs/DESIGN.md §4.7).

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>

namespace vf::detail {

inline constexpr std::size_t kCacheLineAlignment = 64;

template <std::size_t Alignment>
  requires(std::has_single_bit(Alignment))
struct AlignedDeleter {
  template <class T>
  void operator()(T* ptr) const noexcept {
    ::operator delete(static_cast<void*>(ptr), std::align_val_t{Alignment});
  }
};

template <class T, std::size_t Alignment = kCacheLineAlignment>
  requires(std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T> &&
           std::has_single_bit(Alignment) && Alignment >= alignof(T))
using AlignedArray =
    std::unique_ptr<T[], AlignedDeleter<Alignment>>;  // NOLINT(modernize-avoid-c-arrays)

// Allocates `count` uninitialised elements aligned to `Alignment`. Returns nullptr for count == 0.
// T must be an implicit-lifetime type: operator new implicitly creates the array objects (C++20).
template <class T, std::size_t Alignment = kCacheLineAlignment>
  requires(std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T> &&
           std::has_single_bit(Alignment) && Alignment >= alignof(T))
[[nodiscard]] AlignedArray<T, Alignment> make_aligned_array(std::size_t count) {
  if (count == 0) {
    return AlignedArray<T, Alignment>{};
  }
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    throw std::bad_array_new_length();
  }
  void* raw = ::operator new(count * sizeof(T), std::align_val_t{Alignment});
  return AlignedArray<T, Alignment>{static_cast<T*>(raw)};
}

// As make_aligned_array, with every element value-initialised (zero for arithmetic types).
template <class T, std::size_t Alignment = kCacheLineAlignment>
[[nodiscard]] AlignedArray<T, Alignment> make_aligned_array_zeroed(std::size_t count) {
  AlignedArray<T, Alignment> array = make_aligned_array<T, Alignment>(count);
  std::fill_n(array.get(), count, T{});
  return array;
}

// Over-aligned array of std::atomic<T> (not trivially copyable, but trivially destructible), with
// every element constructed holding T{}.
template <class T, std::size_t Alignment = kCacheLineAlignment>
  requires(std::is_integral_v<T> && std::is_trivially_destructible_v<std::atomic<T>> &&
           std::has_single_bit(Alignment) && Alignment >= alignof(std::atomic<T>))
// NOLINTNEXTLINE(modernize-avoid-c-arrays): unique_ptr<T[]> owns the aligned array
using AtomicArray = std::unique_ptr<std::atomic<T>[], AlignedDeleter<Alignment>>;

template <class T, std::size_t Alignment = kCacheLineAlignment>
[[nodiscard]] AtomicArray<T, Alignment> make_atomic_array(std::size_t count) {
  if (count == 0) {
    return AtomicArray<T, Alignment>{};
  }
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(std::atomic<T>)) {
    throw std::bad_array_new_length();
  }
  void* raw = ::operator new(count * sizeof(std::atomic<T>), std::align_val_t{Alignment});
  auto* first = static_cast<std::atomic<T>*>(raw);
  for (std::size_t i = 0; i < count; ++i) {
    ::new (static_cast<void*>(first + i)) std::atomic<T>(T{});  // constructors cannot throw
  }
  return AtomicArray<T, Alignment>{first};
}

// Standard allocator with a minimum alignment, for std::vector<float, AlignedAllocator<float>>.
template <class T, std::size_t Alignment = kCacheLineAlignment>
  requires(std::has_single_bit(Alignment) && Alignment >= alignof(T))
class AlignedAllocator {
 public:
  using value_type = T;
  using is_always_equal = std::true_type;
  using propagate_on_container_move_assignment = std::true_type;

  template <class U>
  struct rebind {  // NOLINT(readability-identifier-naming): name required by the Allocator API
    using other = AlignedAllocator<U, Alignment>;
  };

  constexpr AlignedAllocator() noexcept = default;
  // Implicit by design: containers rebind allocators via converting construction.
  template <class U>
  constexpr AlignedAllocator(  // NOLINT(google-explicit-constructor)
      const AlignedAllocator<U, Alignment>& /*other*/) noexcept {}

  [[nodiscard]] T* allocate(std::size_t count) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::bad_array_new_length();
    }
    return static_cast<T*>(::operator new(count * sizeof(T), std::align_val_t{Alignment}));
  }

  void deallocate(T* ptr, std::size_t /*count*/) noexcept {
    ::operator delete(static_cast<void*>(ptr), std::align_val_t{Alignment});
  }

  friend constexpr bool operator==(const AlignedAllocator&, const AlignedAllocator&) noexcept {
    return true;
  }
};

}  // namespace vf::detail
