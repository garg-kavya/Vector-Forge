// Top-k selection strategies over a stream of n distances with ascending ids, mirroring the Flat
// scan (docs/DESIGN.md §9.13). Used to choose the top-k strategy of FlatBackend.
//
// Patterns: "random" (uniform distances: few candidates are accepted) and "descending" (every
// candidate improves on the current worst: adversarial for insertion).
// Strategies: insertion (SortedInsertionTopK), heap (BoundedMaxHeap), nth_element and partial_sort
// over a full (distance, id) array (these need O(n) scratch, so they are reference points only).

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <utility>
#include <vector>

#include "core/rng.hpp"
#include "search/heaps.hpp"

namespace {

using vf::detail::ScoredId;

// Alternative evaluated against BoundedMaxHeap: sorted array, O(k) shift per accepted candidate.
// Not used by the library (the heap was faster in every measured case; see the results README).
template <vf::detail::SearchCandidate T>
class SortedInsertionTopK {
 public:
  explicit SortedInsertionTopK(std::span<T> storage) noexcept : data_(storage) {}
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] bool full() const noexcept { return size_ == data_.size(); }
  [[nodiscard]] const T& worst() const noexcept { return data_[size_ - 1]; }

  void push(const T& candidate) noexcept {
    std::size_t pos = size_;
    if (size_ == data_.size()) {
      if (size_ == 0 || !vf::detail::candidate_less(candidate, data_[size_ - 1])) {
        return;
      }
      pos = size_ - 1;
    } else {
      ++size_;
    }
    while (pos > 0 && vf::detail::candidate_less(candidate, data_[pos - 1])) {
      data_[pos] = data_[pos - 1];
      --pos;
    }
    data_[pos] = candidate;
  }

 private:
  std::span<T> data_;
  std::size_t size_ = 0;
};

enum class Strategy : std::uint8_t { Insertion, Heap, NthElement, PartialSort };

const std::vector<float>& distances(std::size_t n, bool descending) {
  static std::map<std::pair<std::size_t, bool>, std::vector<float>> cache;
  auto& v = cache[{n, descending}];
  if (v.empty()) {
    v.resize(n);
    vf::detail::Xoshiro256ss rng(n);
    for (std::size_t i = 0; i < n; ++i) {
      v[i] = descending ? static_cast<float>(n - i) : vf::detail::uniform_float(rng(), 0.0F, 1.0F);
    }
  }
  return v;
}

template <class Selector, class WorstFn>
std::size_t stream_select(const std::vector<float>& d, Selector& selector, WorstFn worst) {
  for (std::size_t i = 0; i < d.size(); ++i) {
    if (selector.full() && !(d[i] < worst(selector))) {
      continue;
    }
    selector.push(ScoredId{d[i], static_cast<vf::InternalId>(i)});
  }
  return selector.size();
}

void run(benchmark::State& state, Strategy strategy) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto k = static_cast<std::size_t>(state.range(1));
  const bool descending = state.range(2) != 0;
  const std::vector<float>& d = distances(n, descending);
  std::vector<ScoredId> out(k);
  std::vector<ScoredId> scratch;
  if (strategy == Strategy::NthElement || strategy == Strategy::PartialSort) {
    scratch.resize(n);
  }

  for (auto _ : state) {
    std::size_t count = 0;
    switch (strategy) {
      case Strategy::Insertion: {
        SortedInsertionTopK<ScoredId> sel(out);
        stream_select(d, sel, [](const auto& s) { return s.worst().distance; });
        count = sel.size();
        break;
      }
      case Strategy::Heap: {
        vf::detail::BoundedMaxHeap<ScoredId> sel(out);
        stream_select(d, sel, [](const auto& s) { return s.top().distance; });
        count = sel.sort_ascending();
        break;
      }
      case Strategy::NthElement: {
        for (std::size_t i = 0; i < n; ++i) {
          scratch[i] = {d[i], static_cast<vf::InternalId>(i)};
        }
        const auto less = [](const ScoredId& a, const ScoredId& b) {
          return vf::detail::candidate_less(a, b);
        };
        const auto kth = scratch.begin() + static_cast<std::ptrdiff_t>(std::min(k, n));
        std::nth_element(scratch.begin(), kth, scratch.end(), less);
        std::sort(scratch.begin(), kth, less);
        count = std::min(k, n);
        break;
      }
      case Strategy::PartialSort: {
        for (std::size_t i = 0; i < n; ++i) {
          scratch[i] = {d[i], static_cast<vf::InternalId>(i)};
        }
        const auto kth = scratch.begin() + static_cast<std::ptrdiff_t>(std::min(k, n));
        std::partial_sort(
            scratch.begin(), kth, scratch.end(),
            [](const ScoredId& a, const ScoredId& b) { return vf::detail::candidate_less(a, b); });
        count = std::min(k, n);
        break;
      }
    }
    benchmark::DoNotOptimize(count);
  }
  state.SetLabel(descending ? "descending" : "random");
}

void add_args(benchmark::Benchmark* b) {
  for (const std::int64_t n : {10000, 1000000}) {
    for (const std::int64_t k : {1, 10, 32, 64, 100, 1000}) {
      for (const std::int64_t pattern : {0, 1}) {
        b->Args({n, k, pattern});
      }
    }
  }
  b->ArgNames({"n", "k", "descending"});
}

void BenchTopKInsertion(benchmark::State& s) {
  run(s, Strategy::Insertion);
}
void BenchTopKHeap(benchmark::State& s) {
  run(s, Strategy::Heap);
}
void BenchTopKNthElement(benchmark::State& s) {
  run(s, Strategy::NthElement);
}
void BenchTopKPartialSort(benchmark::State& s) {
  run(s, Strategy::PartialSort);
}

BENCHMARK(BenchTopKInsertion)->Apply(add_args);
BENCHMARK(BenchTopKHeap)->Apply(add_args);
BENCHMARK(BenchTopKNthElement)->Apply(add_args);
BENCHMARK(BenchTopKPartialSort)->Apply(add_args);

}  // namespace
