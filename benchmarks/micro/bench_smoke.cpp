#include <benchmark/benchmark.h>

#include <string>

#include "bench_context.hpp"
#include "core/build_info.hpp"

namespace vf::bench {

void add_extra_context() {
}

}  // namespace vf::bench

namespace {

// Harness sanity check only; not a performance measurement of VectorForge.
void BenchSmokeBuildSummary(benchmark::State& state) {
  for (auto _ : state) {
    std::string s = vf::detail::build_summary();
    benchmark::DoNotOptimize(s.data());
  }
}
BENCHMARK(BenchSmokeBuildSummary);

}  // namespace
