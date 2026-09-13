#include <benchmark/benchmark.h>

#include <string>

#include <vectorforge/simd.hpp>

#include "bench_context.hpp"
#include "core/build_info.hpp"
#include "simd/cpu_features.hpp"

namespace vf::bench {

void add_extra_context() {
  const vf::detail::CpuFeatures& f = vf::detail::cpu_features();
  benchmark::AddCustomContext("vf_cpu_brand", f.brand);
  benchmark::AddCustomContext("vf_cpu_avx2", f.avx2 ? "1" : "0");
  benchmark::AddCustomContext("vf_cpu_fma", f.fma ? "1" : "0");
  benchmark::AddCustomContext("vf_cpu_avx512f", f.avx512f ? "1" : "0");
  benchmark::AddCustomContext("vf_simd_level", std::string(vf::to_string(vf::active_simd_level())));
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
