#pragma once

// Allocation-free priority queues for search candidates (docs/DESIGN.md §9.4, §9.13).
//
// Candidates are ordered lexicographically by (distance, id). Distances are never NaN (inputs are
// validated at the API boundary), so this is a strict weak ordering and results are deterministic
// even when distances tie.

#include <concepts>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include <vectorforge/types.hpp>

#include "core/assert.hpp"

namespace vf::detail {

template <class T>
concept SearchCandidate = requires(const T& c) {
  { c.distance } -> std::convertible_to<float>;
  { c.id } -> std::convertible_to<std::uint64_t>;
};

// Compact (distance, internal id) pair used inside search algorithms.
struct ScoredId {
  float distance = 0.0F;
  InternalId id = kInvalidInternalId;

  friend bool operator==(const ScoredId&, const ScoredId&) = default;
};

template <SearchCandidate T>
[[nodiscard]] constexpr bool candidate_less(const T& a, const T& b) noexcept {
  return a.distance < b.distance || (a.distance == b.distance && a.id < b.id);
}

// Max-heap (worst candidate on top) holding at most storage.size() elements in caller-provided
// storage. Used for top-k selection: push() keeps the best `capacity` candidates seen so far.
template <SearchCandidate T>
class BoundedMaxHeap {
 public:
  explicit BoundedMaxHeap(std::span<T> storage) noexcept : data_(storage) {}

  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] std::size_t capacity() const noexcept { return data_.size(); }
  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
  [[nodiscard]] bool full() const noexcept { return size_ == data_.size(); }

  // Worst retained candidate. Precondition: !empty().
  [[nodiscard]] const T& top() const noexcept {
    VF_ASSERT(size_ > 0, "BoundedMaxHeap::top on empty heap");
    return data_[0];
  }

  // Inserts `candidate` if the heap is not full, or replaces the worst element if `candidate` is
  // better. Returns true if the candidate was retained. A zero-capacity heap retains nothing.
  bool push(const T& candidate) noexcept {
    if (size_ < data_.size()) {
      data_[size_] = candidate;
      sift_up(size_);
      ++size_;
      return true;
    }
    if (size_ == 0 || !candidate_less(candidate, data_[0])) {
      return false;
    }
    data_[0] = candidate;
    sift_down(0, size_);
    return true;
  }

  // Removes the worst element. Precondition: !empty().
  void pop() noexcept {
    VF_ASSERT(size_ > 0, "BoundedMaxHeap::pop on empty heap");
    --size_;
    if (size_ > 0) {
      data_[0] = data_[size_];
      sift_down(0, size_);
    }
  }

  // Sorts the retained elements ascending in place (storage[0..size())) using heapsort and returns
  // the count. The heap is left empty; the sorted elements remain readable in the storage.
  std::size_t sort_ascending() noexcept {
    const std::size_t count = size_;
    for (std::size_t end = count; end > 1; --end) {
      std::swap(data_[0], data_[end - 1]);
      sift_down(0, end - 1);
    }
    size_ = 0;
    return count;
  }

  void clear() noexcept { size_ = 0; }

 private:
  void sift_up(std::size_t index) noexcept {
    while (index > 0) {
      const std::size_t parent = (index - 1) / 2;
      if (!candidate_less(data_[parent], data_[index])) {
        return;
      }
      std::swap(data_[parent], data_[index]);
      index = parent;
    }
  }

  void sift_down(std::size_t index, std::size_t count) noexcept {
    for (;;) {
      const std::size_t left = (2 * index) + 1;
      if (left >= count) {
        return;
      }
      std::size_t largest = left;
      const std::size_t right = left + 1;
      if (right < count && candidate_less(data_[left], data_[right])) {
        largest = right;
      }
      if (!candidate_less(data_[index], data_[largest])) {
        return;
      }
      std::swap(data_[index], data_[largest]);
      index = largest;
    }
  }

  std::span<T> data_;
  std::size_t size_ = 0;
};

// Min-heap (best candidate on top) over reusable vector storage; used as the HNSW candidate queue.
// Allocates only when growing beyond previously reserved capacity.
template <SearchCandidate T>
class MinHeap {
 public:
  void reserve(std::size_t capacity) { data_.reserve(capacity); }
  [[nodiscard]] std::size_t capacity() const noexcept { return data_.capacity(); }
  [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }
  [[nodiscard]] bool empty() const noexcept { return data_.empty(); }
  void clear() noexcept { data_.clear(); }

  // Best candidate. Precondition: !empty().
  [[nodiscard]] const T& top() const noexcept {
    VF_ASSERT(!data_.empty(), "MinHeap::top on empty heap");
    return data_.front();
  }

  void push(const T& candidate) {
    data_.push_back(candidate);
    std::size_t index = data_.size() - 1;
    while (index > 0) {
      const std::size_t parent = (index - 1) / 2;
      if (!candidate_less(data_[index], data_[parent])) {
        break;
      }
      std::swap(data_[parent], data_[index]);
      index = parent;
    }
  }

  // Removes the best candidate. Precondition: !empty().
  void pop() noexcept {
    VF_ASSERT(!data_.empty(), "MinHeap::pop on empty heap");
    data_.front() = data_.back();
    data_.pop_back();
    const std::size_t count = data_.size();
    std::size_t index = 0;
    for (;;) {
      const std::size_t left = (2 * index) + 1;
      if (left >= count) {
        return;
      }
      std::size_t smallest = left;
      const std::size_t right = left + 1;
      if (right < count && candidate_less(data_[right], data_[left])) {
        smallest = right;
      }
      if (!candidate_less(data_[smallest], data_[index])) {
        return;
      }
      std::swap(data_[index], data_[smallest]);
      index = smallest;
    }
  }

 private:
  std::vector<T> data_;
};

}  // namespace vf::detail
