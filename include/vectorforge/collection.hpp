#pragma once

// Collection: the primary entry point of the VectorForge library.
//
// A collection stores float32 vectors of a fixed dimension under user-chosen 64-bit ids and answers
// k-nearest-neighbour queries under its metric (docs/DESIGN.md §8.1).
//
// Thread safety (Phase 2): thread-compatible. Const member functions may be called concurrently
// with each other; non-const member functions require exclusive access (no concurrent readers or
// writers). Built-in synchronisation arrives in Phase 6.
//
// Supported index types: IndexType::Flat (exact search). IndexType::Hnsw arrives in Phase 3;
// create() reports FailedPrecondition for it until then.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <vectorforge/config.hpp>
#include <vectorforge/simd.hpp>
#include <vectorforge/status.hpp>
#include <vectorforge/types.hpp>

namespace vf {

struct MemoryUsage {
  std::size_t vectors_bytes = 0;          // allocated vector chunks
  std::size_t labels_bytes = 0;           // internal -> external id array
  std::size_t id_map_bytes_estimate = 0;  // external -> internal hash map (estimated)
  std::size_t tombstone_bytes = 0;
  std::size_t index_bytes = 0;  // index structures (0 for Flat)

  [[nodiscard]] std::size_t total_bytes() const noexcept {
    return vectors_bytes + labels_bytes + id_map_bytes_estimate + tombstone_bytes + index_bytes;
  }
};

struct CollectionStats {
  std::uint64_t live_count = 0;     // searchable vectors
  std::uint64_t deleted_count = 0;  // tombstoned rows (removed or replaced by upsert)
  std::uint64_t row_count = 0;      // stored rows = live + deleted
  std::uint32_t dim = 0;
  Metric metric = Metric::L2;
  IndexType index = IndexType::Flat;
  bool normalized = false;
  SimdLevel simd = SimdLevel::Scalar;
  MemoryUsage memory;
};

class Collection {
 public:
  // Errors: InvalidArgument (invalid config); FailedPrecondition (index type not yet available).
  [[nodiscard]] static Result<std::unique_ptr<Collection>> create(const CollectionConfig& config);

  Collection(const Collection&) = delete;
  Collection& operator=(const Collection&) = delete;
  Collection(Collection&&) = delete;
  Collection& operator=(Collection&&) = delete;
  ~Collection();

  // Inserts one vector (normalised first if config().effective_normalize()).
  // Errors: DimensionMismatch; InvalidArgument (non-finite or |x| > kMaxAbsComponent, zero vector
  // in a normalised collection, id == kInvalidExternalId); AlreadyExists (id present and
  // !options.upsert); ResourceExhausted (collection full). Strong guarantee: on error nothing
  // changes.
  [[nodiscard]] Status add(ExternalId id, std::span<const float> vector,
                           InsertOptions options = {});

  // Inserts ids.size() vectors given row-major in `rows`. The whole batch is validated before any
  // mutation (sizes, values, duplicate ids within the batch, existing ids unless upsert), so
  // validation errors leave the collection unchanged. Returns the number of vectors inserted.
  // A failure after validation (index error) returns the error; rows before it stay inserted.
  // std::bad_alloc propagates with the same partial-insert semantics.
  [[nodiscard]] Result<std::size_t> add_batch(std::span<const ExternalId> ids,
                                              std::span<const float> rows,
                                              InsertOptions options = {});

  // Tombstones a vector. Errors: NotFound.
  [[nodiscard]] Status remove(ExternalId id);

  // Copy of the stored vector (normalised for normalised collections). Errors: NotFound.
  [[nodiscard]] Result<std::vector<float>> get(ExternalId id) const;
  [[nodiscard]] bool contains(ExternalId id) const noexcept;

  // k nearest neighbours of `query`, ascending by distance (ties: older insertions first).
  // Errors: DimensionMismatch; InvalidArgument (params, non-finite query, zero query in a
  // normalised collection).
  [[nodiscard]] Result<std::vector<Neighbor>> search(std::span<const float> query,
                                                     const SearchParams& params = {}) const;

  // As search(), writing into caller storage: requires out.size() >= params.k and returns the
  // number of results written. Performs no heap allocation on success.
  [[nodiscard]] Result<std::size_t> search_into(std::span<const float> query,
                                                const SearchParams& params,
                                                std::span<Neighbor> out) const;

  // Searches nq row-major queries. For query i, slots [i*k, (i+1)*k) of out_ids/out_distances
  // receive the results; counts[i] is the number of real results and unused slots are padded with
  // kInvalidExternalId / +infinity. All queries are validated before any output is written.
  [[nodiscard]] Status search_batch(std::span<const float> queries, std::size_t nq,
                                    const SearchParams& params, std::span<ExternalId> out_ids,
                                    std::span<float> out_distances,
                                    std::span<std::uint32_t> counts) const;

  // Number of searchable vectors.
  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] CollectionStats stats() const;
  [[nodiscard]] const CollectionConfig& config() const noexcept;

 private:
  struct Impl;
  explicit Collection(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vf
