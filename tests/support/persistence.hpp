#pragma once

// Helpers for persistence tests: building representative collections and comparing them.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <system_error>
#include <vector>

#include <vectorforge/collection.hpp>

#include "collection/collection_factory.hpp"
#include "core/rng.hpp"
#include "storage/binary_io.hpp"
#include "support/test_data.hpp"

namespace vf::test {

inline std::vector<std::byte> read_bytes(const std::filesystem::path& path) {
  std::error_code ec;
  const std::uintmax_t size = std::filesystem::file_size(path, ec);
  std::vector<std::byte> out(ec ? 0 : static_cast<std::size_t>(size));
  std::ifstream in(path, std::ios::binary);
  in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()));
  return out;
}

inline void write_bytes(const std::filesystem::path& path, std::span<const std::byte> bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
}

inline std::vector<std::byte> save_to_memory(const Collection& c) {
  detail::MemorySink sink;
  const Status st = detail::CollectionFactory::save_to(c, sink);
  EXPECT_TRUE(st.ok()) << st.to_string();
  return sink.take();
}

inline CollectionConfig small_config(IndexType index, Metric metric, bool normalize,
                                     std::uint32_t dim) {
  CollectionConfig cfg;
  cfg.dim = dim;
  cfg.metric = metric;
  cfg.normalize = normalize;
  cfg.index = index;
  cfg.hnsw.M = 8;
  cfg.hnsw.ef_construction = 48;
  cfg.hnsw.ef_search = 32;
  cfg.hnsw.seed = 777;
  return cfg;
}

// Applies a deterministic mix of inserts, upserts and removals (ids 1000 + 3i).
inline void apply_operations(Collection& c, std::uint64_t seed, std::size_t inserts,
                             std::uint64_t first_row) {
  const std::uint32_t dim = c.config().dim;
  const ClusteredData data(5, dim, 12, 0.3F);
  const std::vector<float> rows = data.rows(seed, inserts);
  for (std::size_t i = 0; i < inserts; ++i) {
    const ExternalId id = 1000 + (3 * (first_row + i));
    ASSERT_TRUE(c.add(id, std::span<const float>(rows).subspan(i * dim, dim)).ok());
  }
  detail::Xoshiro256ss rng((seed * 17) + 1);
  const std::uint64_t total = first_row + inserts;
  for (std::size_t i = 0; i < inserts / 8; ++i) {
    const ExternalId id = 1000 + (3 * detail::uniform_below(rng, total));
    std::vector<float> v = random_vector(rng, dim);
    ASSERT_TRUE(c.add(id, v, {.upsert = true}).ok());
  }
  for (std::size_t i = 0; i < inserts / 10; ++i) {
    const ExternalId id = 1000 + (3 * detail::uniform_below(rng, total));
    static_cast<void>(c.remove(id));  // may already be removed
  }
}

inline void expect_same_config(const CollectionConfig& a, const CollectionConfig& b) {
  EXPECT_EQ(a.dim, b.dim);
  EXPECT_EQ(a.metric, b.metric);
  EXPECT_EQ(a.normalize, b.normalize);
  EXPECT_EQ(a.index, b.index);
  EXPECT_EQ(a.hnsw.M, b.hnsw.M);
  EXPECT_EQ(a.hnsw.ef_construction, b.hnsw.ef_construction);
  EXPECT_EQ(a.hnsw.ef_search, b.hnsw.ef_search);
  EXPECT_EQ(a.hnsw.max_level, b.hnsw.max_level);
  EXPECT_EQ(a.hnsw.seed, b.hnsw.seed);
}

// Same answers for the same queries: ids, order and bit-identical distances.
inline void expect_same_results(const Collection& a, const Collection& b,
                                std::span<const float> queries, std::uint32_t k) {
  const std::uint32_t dim = a.config().dim;
  SearchParams p;
  p.k = k;
  p.ef_search = 64;
  for (std::size_t q = 0; q < queries.size() / dim; ++q) {
    const auto query = queries.subspan(q * dim, dim);
    const auto ra = a.search(query, p);
    const auto rb = b.search(query, p);
    ASSERT_TRUE(ra.ok() && rb.ok());
    ASSERT_EQ(ra.value(), rb.value()) << "query " << q;
  }
}

}  // namespace vf::test
