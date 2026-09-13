// Exact search latency baseline (docs/DESIGN.md §21 Phase 2): one query at a time through
// Collection::search_into on a Flat collection, single thread, active kernel tier (scalar in Phase
// 2). Data: uniform random vectors in [-1, 1) (distribution barely matters for a full scan).

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <vectorforge/collection.hpp>

#include "core/rng.hpp"

namespace {

constexpr std::size_t kQueries = 64;
constexpr std::uint32_t kK = 10;

struct Dataset {
  std::unique_ptr<vf::Collection> collection;
  std::vector<float> queries;
};

const Dataset& dataset(std::size_t n, std::uint32_t dim, vf::Metric metric) {
  static std::map<std::tuple<std::size_t, std::uint32_t, vf::Metric>, Dataset> cache;
  Dataset& ds = cache[{n, dim, metric}];
  if (!ds.collection) {
    vf::CollectionConfig cfg;
    cfg.dim = dim;
    cfg.metric = metric;
    cfg.index = vf::IndexType::Flat;
    ds.collection = vf::Collection::create(cfg).value();
    vf::detail::Xoshiro256ss rng(n * 131U + dim);
    std::vector<float> rows(n * dim);
    for (float& x : rows) {
      x = vf::detail::uniform_float(rng(), -1.0F, 1.0F);
    }
    std::vector<vf::ExternalId> ids(n);
    for (std::size_t i = 0; i < n; ++i) {
      ids[i] = i;
    }
    if (!ds.collection->add_batch(ids, rows).ok()) {
      std::abort();
    }
    ds.queries.resize(kQueries * dim);
    for (float& x : ds.queries) {
      x = vf::detail::uniform_float(rng(), -1.0F, 1.0F);
    }
  }
  return ds;
}

void run(benchmark::State& state, vf::Metric metric) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto dim = static_cast<std::uint32_t>(state.range(1));
  const Dataset& ds = dataset(n, dim, metric);
  std::vector<vf::Neighbor> out(kK);
  vf::SearchParams params;
  params.k = kK;
  std::size_t q = 0;
  for (auto _ : state) {
    const auto query = std::span<const float>(ds.queries).subspan(q * dim, dim);
    benchmark::DoNotOptimize(ds.collection->search_into(query, params, out));
    q = (q + 1) % kQueries;
  }
  state.counters["qps"] = benchmark::Counter(1.0, benchmark::Counter::kIsIterationInvariantRate);
  state.SetLabel(std::string(vf::to_string(metric)) +
                 " k=10 simd=" + std::string(vf::to_string(vf::active_simd_level())));
}

void add_args(benchmark::Benchmark* b) {
  for (const std::int64_t n : {10000, 100000}) {
    for (const std::int64_t dim : {128, 768}) {
      b->Args({n, dim});
    }
  }
  b->ArgNames({"n", "dim"});
}

void BenchFlatSearchL2(benchmark::State& s) {
  run(s, vf::Metric::L2);
}
void BenchFlatSearchCosine(benchmark::State& s) {
  run(s, vf::Metric::Cosine);
}

BENCHMARK(BenchFlatSearchL2)->Apply(add_args)->Unit(benchmark::kMillisecond);
BENCHMARK(BenchFlatSearchCosine)->Apply(add_args)->Unit(benchmark::kMillisecond);

}  // namespace
