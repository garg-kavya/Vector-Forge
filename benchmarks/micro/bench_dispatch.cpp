// Dispatch-level experiment (docs/DESIGN.md §10.4, docs/simd.md "Dispatch cost").
//
// Question: does calling kernels through the runtime-selected table cost enough at d = 128 to
// justify instantiating the whole search loop per instruction set? Three ways to compute the same
// distances from one query to rows visited in random order (the HNSW access pattern):
//
//   Table     query_distance(*table, ScoreMode::L2, ...) through a KernelTable pointer, exactly as
//             FlatBackend/HnswBackend do (switch on the score mode + indirect call)
//   Direct    a direct call to the extern avx2::l2sq_acc4 (no indirection, still a call)
//   Inlined   the acc4 kernel inlined into an AVX2-compiled loop (per-ISA instantiation)
//
// All three read the same precomputed row pointers, so memory access is identical. Args: dim,
// rows (1 024 = cache-hot, 100 000 = memory-bound). Counter "s_per_distance" (seconds).

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "core/aligned_alloc.hpp"
#include "core/rng.hpp"
#include "index/index_backend.hpp"
#include "index/query_distance.hpp"
#include "simd/cpu_features.hpp"
#include "simd/dispatch.hpp"
#include "simd/kernels.hpp"

#if defined(VF_BENCH_HAVE_AVX2_TU)
#include "bench_dispatch.hpp"
#endif

namespace {

constexpr std::size_t kLookups = 4096;

struct Workload {
  std::size_t dim = 0;
  vf::detail::AlignedArray<float> data;
  std::vector<float> query;
  std::vector<const float*> rows;  // kLookups random rows
};

Workload make_workload(std::size_t dim, std::size_t row_count) {
  Workload w;
  w.dim = dim;
  vf::detail::Xoshiro256ss rng(dim + row_count);
  w.data = vf::detail::make_aligned_array<float>(dim * row_count);
  for (std::size_t i = 0; i < dim * row_count; ++i) {
    w.data[i] = vf::detail::uniform_float(rng(), -1.0F, 1.0F);
  }
  w.query.resize(dim);
  for (float& x : w.query) {
    x = vf::detail::uniform_float(rng(), -1.0F, 1.0F);
  }
  w.rows.resize(kLookups);
  for (const float*& row : w.rows) {
    row = w.data.get() + ((rng() % row_count) * dim);
  }
  return w;
}

// Mirrors the backends: the table is reached through a stored pointer.
struct TableUser {
  const vf::detail::KernelTable* kernels;
  vf::detail::ScoreMode mode;
};

enum class Variant : std::uint8_t { Table, Direct, Inlined };

void run(benchmark::State& state, const TableUser& user, Variant variant) {
  const auto dim = static_cast<std::size_t>(state.range(0));
  const Workload w = make_workload(dim, static_cast<std::size_t>(state.range(1)));
  const vf::detail::QueryView view{.data = w.query.data(), .inv_norm = 1.0F};
  for (auto _ : state) {
    float sum = 0.0F;
    switch (variant) {
      case Variant::Table:
        for (const float* row : w.rows) {
          sum += vf::detail::query_distance(*user.kernels, user.mode, view, row, dim);
        }
        break;
      case Variant::Direct:
        for (const float* row : w.rows) {
          sum += vf::detail::avx2::l2sq_acc4(view.data, row, dim);
        }
        break;
      case Variant::Inlined:
#if defined(VF_BENCH_HAVE_AVX2_TU)
        sum = vf::bench::inlined_avx2_l2sq_sum(view.data, w.rows.data(), w.rows.size(), dim);
#endif
        break;
    }
    benchmark::DoNotOptimize(sum);
  }
  state.counters["s_per_distance"] = benchmark::Counter(
      static_cast<double>(kLookups),
      benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
}

bool register_dispatch_benchmarks() {
#if defined(VF_BENCH_HAVE_AVX2_TU)
  const vf::detail::KernelTable* avx2 = vf::detail::avx2_kernel_table(vf::detail::cpu_features());
  if (avx2 == nullptr || avx2->l2sq != &vf::detail::avx2::l2sq_acc4) {
    return false;  // experiment compares against acc4; nothing to measure here
  }
  static const TableUser kUser{.kernels = avx2, .mode = vf::detail::ScoreMode::L2};
  for (const auto& [name, variant] :
       {std::pair{"Table", Variant::Table}, std::pair{"Direct", Variant::Direct},
        std::pair{"Inlined", Variant::Inlined}}) {
    benchmark::RegisterBenchmark(std::string("BenchDispatch/") + name,
                                 [variant](benchmark::State& s) { run(s, kUser, variant); })
        ->Args({128, 1024})
        ->Args({128, 100000})
        ->Args({768, 1024})
        ->Args({768, 100000})
        ->ArgNames({"dim", "rows"});
  }
  return true;
#else
  return false;
#endif
}

[[maybe_unused]] const bool kRegistered = register_dispatch_benchmarks();

}  // namespace
