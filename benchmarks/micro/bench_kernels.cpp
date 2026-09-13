// Distance kernel micro benchmarks (docs/DESIGN.md §10.6).
//
// Phase 1 baseline: scalar reference vs compiler auto-vectorised variants. Inputs are cache-hot
// (a single pair of vectors) to isolate compute cost. "aligned" inputs start on a 64-byte boundary;
// "offset" inputs start one float later, i.e. misaligned with respect to 8/16/32-byte SIMD widths.
// Counter "dims_per_s" = vector components processed per second.

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "core/aligned_alloc.hpp"
#include "core/rng.hpp"
#include "simd/kernels.hpp"

namespace {

using vf::detail::KernelTable;

constexpr std::size_t kBatchRows = 1024;

struct PairInputs {
  vf::detail::AlignedArray<float> a_buf;
  vf::detail::AlignedArray<float> b_buf;
  float* a = nullptr;
  float* b = nullptr;
};

PairInputs make_pair_inputs(std::size_t dim, bool offset) {
  PairInputs in;
  in.a_buf = vf::detail::make_aligned_array<float>(dim + 1);
  in.b_buf = vf::detail::make_aligned_array<float>(dim + 1);
  in.a = in.a_buf.get() + (offset ? 1 : 0);
  in.b = in.b_buf.get() + (offset ? 1 : 0);
  vf::detail::Xoshiro256ss rng(dim);
  for (std::size_t i = 0; i < dim; ++i) {
    in.a[i] = vf::detail::uniform_float(rng(), -1.0F, 1.0F);
    in.b[i] = vf::detail::uniform_float(rng(), -1.0F, 1.0F);
  }
  return in;
}

enum class Op : std::uint8_t { Dot, L2sq };

void run_pair(benchmark::State& state, const KernelTable& table, Op op) {
  const auto dim = static_cast<std::size_t>(state.range(0));
  const bool offset = state.range(1) != 0;
  const PairInputs in = make_pair_inputs(dim, offset);
  const auto kernel = op == Op::Dot ? table.dot : table.l2sq;
  for (auto _ : state) {
    benchmark::DoNotOptimize(kernel(in.a, in.b, dim));
  }
  state.counters["dims_per_s"] =
      benchmark::Counter(static_cast<double>(dim), benchmark::Counter::kIsIterationInvariantRate);
  state.SetLabel(std::string(table.name) + (offset ? " offset" : " aligned"));
}

void run_batch(benchmark::State& state, const KernelTable& table) {
  const auto dim = static_cast<std::size_t>(state.range(0));
  vf::detail::Xoshiro256ss rng(dim * 31U);
  auto query = vf::detail::make_aligned_array<float>(dim);
  auto rows = vf::detail::make_aligned_array<float>(dim * kBatchRows);
  std::vector<float> out(kBatchRows);
  for (std::size_t i = 0; i < dim; ++i) {
    query[i] = vf::detail::uniform_float(rng(), -1.0F, 1.0F);
  }
  for (std::size_t i = 0; i < dim * kBatchRows; ++i) {
    rows[i] = vf::detail::uniform_float(rng(), -1.0F, 1.0F);
  }
  for (auto _ : state) {
    table.l2sq_1_to_n(query.get(), rows.get(), kBatchRows, dim, out.data());
    benchmark::DoNotOptimize(out.data());
  }
  state.counters["dims_per_s"] = benchmark::Counter(static_cast<double>(dim * kBatchRows),
                                                    benchmark::Counter::kIsIterationInvariantRate);
  state.SetLabel(std::string(table.name) + " rows=" + std::to_string(kBatchRows));
}

void add_args(benchmark::Benchmark* b) {
  for (const std::int64_t dim : {8, 16, 100, 128, 384, 768, 1536, 1537}) {
    for (const std::int64_t offset : {0, 1}) {
      b->Args({dim, offset});
    }
  }
  b->ArgNames({"dim", "offset"});
}

void BenchDotScalar(benchmark::State& s) {
  run_pair(s, vf::detail::scalar_kernel_table(), Op::Dot);
}
void BenchDotAutovec(benchmark::State& s) {
  run_pair(s, vf::detail::scalar_autovec_kernel_table(), Op::Dot);
}
void BenchL2sqScalar(benchmark::State& s) {
  run_pair(s, vf::detail::scalar_kernel_table(), Op::L2sq);
}
void BenchL2sqAutovec(benchmark::State& s) {
  run_pair(s, vf::detail::scalar_autovec_kernel_table(), Op::L2sq);
}
void BenchL2sqBatchScalar(benchmark::State& s) {
  run_batch(s, vf::detail::scalar_kernel_table());
}
void BenchL2sqBatchAutovec(benchmark::State& s) {
  run_batch(s, vf::detail::scalar_autovec_kernel_table());
}

BENCHMARK(BenchDotScalar)->Apply(add_args);
BENCHMARK(BenchDotAutovec)->Apply(add_args);
BENCHMARK(BenchL2sqScalar)->Apply(add_args);
BENCHMARK(BenchL2sqAutovec)->Apply(add_args);
BENCHMARK(BenchL2sqBatchScalar)->ArgName("dim")->Arg(128)->Arg(768)->Arg(1536);
BENCHMARK(BenchL2sqBatchAutovec)->ArgName("dim")->Arg(128)->Arg(768)->Arg(1536);

}  // namespace
