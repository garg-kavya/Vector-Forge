// Single-threaded HNSW builds are deterministic (docs/DESIGN.md §9.2, §15.3): same data, parameters
// and seed give an identical graph (compared through its canonical encoding) and identical search
// results. The golden fingerprint pins the result across compilers and standard libraries: data,
// level assignment and scalar kernels are all bit-reproducible.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "support/hnsw_fixture.hpp"
#include "support/test_data.hpp"

namespace {

using vf::Metric;
using vf::Neighbor;

constexpr std::uint32_t kDim = 12;
constexpr std::size_t kRows = 2000;

std::uint64_t fnv1a(const std::vector<std::uint8_t>& bytes) {
  std::uint64_t h = 0xCBF29CE484222325ULL;
  for (const std::uint8_t b : bytes) {
    h ^= b;
    h *= 0x100000001B3ULL;
  }
  return h;
}

vf::HnswParams params_with_seed(std::uint64_t seed) {
  vf::HnswParams hp;
  hp.M = 10;
  hp.ef_construction = 80;
  hp.seed = seed;
  return hp;
}

struct Build {
  std::vector<std::uint8_t> graph;
  std::vector<std::vector<Neighbor>> results;
};

Build build(Metric metric, std::uint64_t seed, std::size_t nodes_per_chunk = 0) {
  const vf::test::ClusteredData data(5, kDim, 16, 0.15F);
  vf::test::HnswFixture f(kDim, metric, params_with_seed(seed),
                          {.nodes_per_chunk = nodes_per_chunk});
  f.add_rows(data.rows(1, kRows));
  Build b;
  b.graph = f.backend->graph().canonical_bytes();
  const std::vector<float> queries = data.rows(2, 20);
  for (std::size_t q = 0; q < 20; ++q) {
    b.results.push_back(f.search(std::span<const float>(queries).subspan(q * kDim, kDim), 10, 32));
  }
  return b;
}

TEST(HnswDeterminism, SameSeedSameGraphAndResults) {
  for (const Metric metric : {Metric::L2, Metric::InnerProduct, Metric::Cosine}) {
    SCOPED_TRACE(vf::to_string(metric));
    const Build a = build(metric, 1234);
    const Build b = build(metric, 1234);
    const Build chunked = build(metric, 1234, 64);
    EXPECT_EQ(a.graph, b.graph);
    EXPECT_EQ(a.graph, chunked.graph) << "storage chunking must not influence the graph";
    EXPECT_EQ(a.results, b.results);
    EXPECT_EQ(a.results, chunked.results);
  }
}

TEST(HnswDeterminism, DifferentSeedDifferentGraph) {
  EXPECT_NE(build(Metric::L2, 1234).graph, build(Metric::L2, 4321).graph);
}

TEST(HnswDeterminism, GoldenFingerprint) {
  // Fingerprints of the canonical graph encoding, recorded from the first implementation and
  // identical on MSVC 19.50, GCC 13.3, GCC 15.2 and Clang 18. A change means graphs built by
  // different versions (or platforms) differ; update deliberately and note it in CHANGELOG.md.
  EXPECT_EQ(fnv1a(build(Metric::L2, 1234).graph), 13976385902449391624ULL);
  EXPECT_EQ(fnv1a(build(Metric::Cosine, 1234).graph), 6763529789119153829ULL);
}

}  // namespace
