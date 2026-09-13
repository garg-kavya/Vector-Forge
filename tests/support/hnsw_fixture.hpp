#pragma once

// Owns the storage an HnswBackend indexes, mirroring what Collection does (append, then add), so
// tests can inspect the graph directly.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <vectorforge/config.hpp>
#include <vectorforge/types.hpp>

#include "core/vector_ops.hpp"
#include "index/hnsw/hnsw_backend.hpp"
#include "index/hnsw/hnsw_validator.hpp"
#include "simd/kernels.hpp"
#include "storage/tombstones.hpp"
#include "storage/vector_store.hpp"

namespace vf::test {

struct HnswFixture {
  HnswFixture(std::uint32_t dim, Metric metric_in, const HnswParams& params,
              const detail::HnswBuildOptions& options = {}, std::size_t rows_per_chunk = 0)
      : metric(metric_in),
        normalized(metric_in == Metric::Cosine),
        store(detail::VectorStore::create({.dim = dim, .rows_per_chunk = rows_per_chunk}).value()),
        backend(detail::HnswBackend::create(store, deleted, metric_in, normalized,
                                            detail::kernels(), params, options)
                    .value()) {}

  // The backend points into this object.
  HnswFixture(const HnswFixture&) = delete;
  HnswFixture& operator=(const HnswFixture&) = delete;
  HnswFixture(HnswFixture&&) = delete;
  HnswFixture& operator=(HnswFixture&&) = delete;
  ~HnswFixture() = default;

  // Appends and indexes one row (normalised first for cosine). Returns the internal id.
  InternalId add(std::span<const float> row) {
    std::vector<float> copy(row.begin(), row.end());
    if (normalized) {
      EXPECT_TRUE(detail::normalize_inplace(copy, detail::kernels()).ok());
    }
    const InternalId id = store.append(copy).value();
    deleted.ensure_size(store.size());
    const Status st = backend->add(id);
    EXPECT_TRUE(st.ok()) << st.to_string();
    return id;
  }

  void add_rows(std::span<const float> rows) {
    const std::size_t dim = store.dim();
    for (std::size_t r = 0; r < rows.size() / dim; ++r) {
      add(rows.subspan(r * dim, dim));
    }
  }

  void remove(InternalId id) {
    deleted.set(id);
    backend->remove(id);
  }

  [[nodiscard]] std::vector<Neighbor> search(std::span<const float> query, std::uint32_t k,
                                             std::uint32_t ef = 0) const {
    detail::QueryView view{.data = query.data(), .inv_norm = 1.0F};
    if (normalized) {
      view.inv_norm = detail::inverse_norm(query, detail::kernels()).value();
    }
    std::vector<Neighbor> out(k);
    out.resize(backend->search(view, {.k = k, .ef = ef}, out));
    return out;
  }

  [[nodiscard]] detail::HnswValidator validator() const {
    return detail::HnswValidator(backend->graph());
  }

  Metric metric;
  bool normalized;
  detail::VectorStore store;
  detail::TombstoneSet deleted;
  std::unique_ptr<detail::HnswBackend> backend;
};

}  // namespace vf::test
