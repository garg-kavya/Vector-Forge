// Visited-set strategies for graph search (docs/DESIGN.md §9.5): the epoch-stamped array used by
// HNSW versus clearing a byte array per search and a std::unordered_set. Each iteration simulates
// one search: reset, then `visits` visit() calls on random ids of an n-node graph (about half of
// them repeats).

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

#include "core/rng.hpp"
#include "search/visited_set.hpp"

namespace {

constexpr std::size_t kPatterns = 16;

std::vector<std::uint32_t> make_ids(std::size_t n, std::size_t visits) {
  vf::detail::Xoshiro256ss rng(n * 31U + visits);
  std::vector<std::uint32_t> ids(visits * kPatterns);
  for (std::size_t p = 0; p < kPatterns; ++p) {
    for (std::size_t i = 0; i < visits; ++i) {
      // Revisit an earlier id about half of the time.
      const bool repeat = i > 0 && (rng() & 1U) != 0;
      ids[(p * visits) + i] = repeat
                                  ? ids[(p * visits) + vf::detail::uniform_below(rng, i)]
                                  : static_cast<std::uint32_t>(vf::detail::uniform_below(rng, n));
    }
  }
  return ids;
}

void BenchVisitedEpoch(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto visits = static_cast<std::size_t>(state.range(1));
  const std::vector<std::uint32_t> ids = make_ids(n, visits);
  vf::detail::VisitedSet set;
  set.ensure_capacity(n);
  std::size_t p = 0;
  for (auto _ : state) {
    set.reset();
    std::size_t fresh = 0;
    for (std::size_t i = 0; i < visits; ++i) {
      fresh += set.visit(ids[(p * visits) + i]) ? 1U : 0U;
    }
    benchmark::DoNotOptimize(fresh);
    p = (p + 1) % kPatterns;
  }
}

void BenchVisitedClearBytes(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto visits = static_cast<std::size_t>(state.range(1));
  const std::vector<std::uint32_t> ids = make_ids(n, visits);
  std::vector<std::uint8_t> marks(n);
  std::size_t p = 0;
  for (auto _ : state) {
    std::fill(marks.begin(), marks.end(), std::uint8_t{0});
    std::size_t fresh = 0;
    for (std::size_t i = 0; i < visits; ++i) {
      std::uint8_t& m = marks[ids[(p * visits) + i]];
      fresh += m == 0 ? 1U : 0U;
      m = 1;
    }
    benchmark::DoNotOptimize(fresh);
    p = (p + 1) % kPatterns;
  }
}

void BenchVisitedHashSet(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto visits = static_cast<std::size_t>(state.range(1));
  const std::vector<std::uint32_t> ids = make_ids(n, visits);
  std::unordered_set<std::uint32_t> set;
  set.reserve(visits);
  std::size_t p = 0;
  for (auto _ : state) {
    set.clear();
    std::size_t fresh = 0;
    for (std::size_t i = 0; i < visits; ++i) {
      fresh += set.insert(ids[(p * visits) + i]).second ? 1U : 0U;
    }
    benchmark::DoNotOptimize(fresh);
    p = (p + 1) % kPatterns;
  }
}

void add_args(benchmark::Benchmark* b) {
  for (const std::int64_t n : {10000, 1000000}) {
    for (const std::int64_t visits : {512, 4096}) {
      b->Args({n, visits});
    }
  }
  b->ArgNames({"n", "visits"});
}

BENCHMARK(BenchVisitedEpoch)->Apply(add_args);
BENCHMARK(BenchVisitedClearBytes)->Apply(add_args);
BENCHMARK(BenchVisitedHashSet)->Apply(add_args);

}  // namespace
