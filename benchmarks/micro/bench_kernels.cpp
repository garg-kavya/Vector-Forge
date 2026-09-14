// Distance kernel micro benchmarks (docs/DESIGN.md §10.6, docs/simd.md).
//
// Every kernel table this machine can run: scalar reference, compiler auto-vectorised baseline and
// each AVX2 variant (acc4, acc1, acc4_masked). Benchmarks are registered at startup for the tables
// that exist, named <Kernel>/<table>.
//
//   Dot, L2sq        one pair of cache-hot vectors; args dim, offset (0 = both inputs start on a
//                    64-byte boundary, 1 = both start one float later, misaligned for 8/16/32-byte
//                    SIMD widths)
//   Norm2            one cache-hot vector, aligned
//   L2sqBatchHot     l2sq_1_to_n over 1 024 rows (compute-bound)
//   L2sqBatchCold    l2sq_1_to_n over 100 000 rows (memory-bandwidth-bound at larger dims)
//
// Counter "dims_per_s" = vector components processed per second.

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "core/aligned_alloc.hpp"
#include "core/rng.hpp"
#include "simd/cpu_features.hpp"
#include "simd/dispatch.hpp"
#include "simd/kernels.hpp"

namespace {

using vf::detail::KernelTable;

constexpr std::size_t kHotRows = 1024;
constexpr std::size_t kColdRows = 100000;

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

void run_pair(benchmark::State& state, const KernelTable* table, Op op) {
  const auto dim = static_cast<std::size_t>(state.range(0));
  const bool offset = state.range(1) != 0;
  const PairInputs in = make_pair_inputs(dim, offset);
  const auto kernel = op == Op::Dot ? table->dot : table->l2sq;
  for (auto _ : state) {
    benchmark::DoNotOptimize(kernel(in.a, in.b, dim));
  }
  state.counters["dims_per_s"] =
      benchmark::Counter(static_cast<double>(dim), benchmark::Counter::kIsIterationInvariantRate);
  state.SetLabel(std::string(table->name) + (offset ? " offset" : " aligned"));
}

void run_norm(benchmark::State& state, const KernelTable* table) {
  const auto dim = static_cast<std::size_t>(state.range(0));
  const PairInputs in = make_pair_inputs(dim, false);
  for (auto _ : state) {
    benchmark::DoNotOptimize(table->norm2(in.a, dim));
  }
  state.counters["dims_per_s"] =
      benchmark::Counter(static_cast<double>(dim), benchmark::Counter::kIsIterationInvariantRate);
  state.SetLabel(std::string(table->name));
}

void run_batch(benchmark::State& state, const KernelTable* table, std::size_t rows) {
  const auto dim = static_cast<std::size_t>(state.range(0));
  vf::detail::Xoshiro256ss rng(dim * 31U);
  auto query = vf::detail::make_aligned_array<float>(dim);
  auto data = vf::detail::make_aligned_array<float>(dim * rows);
  std::vector<float> out(rows);
  for (std::size_t i = 0; i < dim; ++i) {
    query[i] = vf::detail::uniform_float(rng(), -1.0F, 1.0F);
  }
  for (std::size_t i = 0; i < dim * rows; ++i) {
    data[i] = vf::detail::uniform_float(rng(), -1.0F, 1.0F);
  }
  for (auto _ : state) {
    table->l2sq_1_to_n(query.get(), data.get(), rows, dim, out.data());
    benchmark::DoNotOptimize(out.data());
  }
  state.counters["dims_per_s"] = benchmark::Counter(static_cast<double>(dim * rows),
                                                    benchmark::Counter::kIsIterationInvariantRate);
  state.SetLabel(std::string(table->name) + " rows=" + std::to_string(rows));
}

std::vector<const KernelTable*> tables() {
  std::vector<const KernelTable*> out = {&vf::detail::scalar_kernel_table(),
                                         &vf::detail::scalar_autovec_kernel_table()};
  for (const KernelTable& t : vf::detail::avx2_variant_tables(vf::detail::cpu_features())) {
    out.push_back(&t);
  }
  return out;
}

bool register_kernel_benchmarks() {
  for (const KernelTable* t : tables()) {
    const std::string suffix = "/" + std::string(t->name);
    for (const Op op : {Op::Dot, Op::L2sq}) {
      const std::string name = (op == Op::Dot ? "BenchDot" : "BenchL2sq") + suffix;
      benchmark::Benchmark* b =
          benchmark::RegisterBenchmark(name, [t, op](benchmark::State& s) { run_pair(s, t, op); });
      for (const std::int64_t dim : {8, 16, 100, 128, 384, 768, 1536, 1537}) {
        for (const std::int64_t offset : {0, 1}) {
          b->Args({dim, offset});
        }
      }
      b->ArgNames({"dim", "offset"});
    }
    benchmark::RegisterBenchmark("BenchNorm2" + suffix,
                                 [t](benchmark::State& s) { run_norm(s, t); })
        ->ArgName("dim")
        ->Arg(128)
        ->Arg(768)
        ->Arg(1536);
    benchmark::RegisterBenchmark("BenchL2sqBatchHot" + suffix,
                                 [t](benchmark::State& s) { run_batch(s, t, kHotRows); })
        ->ArgName("dim")
        ->Arg(128)
        ->Arg(768)
        ->Arg(1536);
    benchmark::RegisterBenchmark("BenchL2sqBatchCold" + suffix,
                                 [t](benchmark::State& s) { run_batch(s, t, kColdRows); })
        ->ArgName("dim")
        ->Arg(128)
        ->Arg(768)
        ->Unit(benchmark::kMillisecond);
  }
  return true;
}

[[maybe_unused]] const bool kRegistered = register_kernel_benchmarks();

}  // namespace
