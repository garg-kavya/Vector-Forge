#pragma once

// Internal interface implemented by index types (Flat now, HNSW from Phase 3).
//
// A backend indexes rows that the owning collection has already appended to its VectorStore and
// reads deletion markers from the collection's TombstoneSet; it owns neither. The virtual call is
// per query/insert, never per distance computation.
//
// Thread safety contract: search() and stats() are const and safe to call concurrently with each
// other; add()/remove() require exclusive access (the Collection enforces this).

#include <cstddef>
#include <cstdint>
#include <span>

#include <vectorforge/status.hpp>
#include <vectorforge/types.hpp>

namespace vf::detail {

// Prepared query. For normalised collections the query is not copied or normalised; instead
// `inv_norm` = 1 / ||query|| scales dot products (see FlatBackend for the exact formulas).
struct QueryView {
  const float* data = nullptr;
  float inv_norm = 1.0F;
};

struct SearchKnobs {
  std::uint32_t k = 10;
  std::uint32_t ef = 0;  // 0 = backend default; ignored by Flat
};

struct BackendStats {
  std::size_t index_bytes = 0;  // memory owned by the backend itself (not vectors/labels)
};

class IndexBackend {
 public:
  IndexBackend() = default;
  IndexBackend(const IndexBackend&) = delete;
  IndexBackend& operator=(const IndexBackend&) = delete;
  IndexBackend(IndexBackend&&) = delete;
  IndexBackend& operator=(IndexBackend&&) = delete;
  virtual ~IndexBackend() = default;

  [[nodiscard]] virtual IndexType type() const noexcept = 0;

  // Indexes row `id`, which has been appended to the vector store (id == store.size() - 1).
  [[nodiscard]] virtual Status add(InternalId id) = 0;

  // Notifies that row `id` was tombstoned.
  virtual void remove(InternalId id) noexcept = 0;

  // Writes up to `knobs.k` best non-tombstoned rows into `out`, sorted by ascending
  // (distance, internal id), with Neighbor::id holding the *internal* id. Returns the count.
  // Preconditions: out.size() >= knobs.k >= 1; query validated.
  [[nodiscard]] virtual std::size_t search(const QueryView& query, const SearchKnobs& knobs,
                                           std::span<Neighbor> out) const noexcept = 0;

  [[nodiscard]] virtual BackendStats stats() const noexcept = 0;
};

}  // namespace vf::detail
