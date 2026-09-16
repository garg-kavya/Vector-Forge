#pragma once

// Collection: the primary entry point of the VectorForge library.
//
// A collection stores float32 vectors of a fixed dimension under user-chosen 64-bit ids and answers
// k-nearest-neighbour queries under its metric (docs/DESIGN.md §8.1).
//
// Thread safety (docs/concurrency.md): every member function may be called concurrently from any
// number of threads. Reads (search*, get, contains, size, stats, save) run in parallel under a
// shared lock. Mutations (add, add_batch, remove) are serialised and block reads only while they
// modify the collection; add_batch holds the lock for at most about 2 ms at a time so that searches
// interleave with long ingestions (the lock is fair: docs/concurrency.md "Fairness"). For HNSW
// with CollectionConfig::concurrency == Concurrent (the default), add_batch holds the lock only to
// append rows and links them into the graph while searches run. compact()
// lets searches continue while it rebuilds and blocks them only for the final swap. A search that
// runs concurrently with remove(id) may or may not return id; a search that starts after
// remove(id) returned never does.
//
// Index types:
//   IndexType::Flat - exact search; results are the true k nearest neighbours.
//   IndexType::Hnsw - approximate search over a Hierarchical Navigable Small World graph
//                     (docs/hnsw.md); results may miss true neighbours, controlled by
//                     HnswParams::ef_construction and SearchParams::ef_search. Removed vectors stay
//                     in the graph for navigation and are never returned.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

#include <vectorforge/config.hpp>
#include <vectorforge/simd.hpp>
#include <vectorforge/status.hpp>
#include <vectorforge/types.hpp>

namespace vf {

class ThreadPool;

namespace detail {
struct CollectionFactory;
}  // namespace detail

struct MemoryUsage {
  std::size_t vectors_bytes = 0;          // allocated heap vector chunks
  std::size_t mapped_vectors_bytes = 0;   // vectors in a memory-mapped file (not in total)
  std::size_t labels_bytes = 0;           // internal -> external id array
  std::size_t id_map_bytes_estimate = 0;  // external -> internal hash map (estimated)
  std::size_t tombstone_bytes = 0;
  std::size_t index_bytes = 0;  // index structures and build scratch (0 for Flat)

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

struct CompactStats {
  std::uint64_t rows_before = 0;   // stored rows before compaction
  std::uint64_t removed_rows = 0;  // tombstoned rows dropped
  std::uint64_t rows_after = 0;    // stored rows after compaction (= live vectors)
};

class Collection {
 public:
  // Errors: InvalidArgument (invalid config, including HnswParams for IndexType::Hnsw).
  [[nodiscard]] static Result<std::unique_ptr<Collection>> create(const CollectionConfig& config);

  // Opens an index file written by save() (docs/storage-format.md). The file is validated as
  // untrusted input: structure always, checksums per options.verify. With options.use_mmap the
  // vectors are served from a read-only mapping (the file must not be modified or truncated while
  // the collection lives; on Windows it cannot be replaced meanwhile); new vectors go to heap
  // memory. Collections smaller than one vector chunk (16 MiB) are copied even with use_mmap.
  // The graph, labels and tombstones are always loaded into memory.
  // Errors: IoError (cannot open or map), CorruptData (invalid or damaged file),
  // UnsupportedVersion (other format major version).
  [[nodiscard]] static Result<std::unique_ptr<Collection>> load(const std::filesystem::path& file,
                                                                const LoadOptions& options = {});

  // Writes the collection to `file` atomically: a temporary file "<file>.tmp" is written, synced
  // and renamed over `file`, so a crash leaves either the old or the new file. Saving the same
  // state twice produces identical bytes. Replacing a file that is currently memory-mapped (for
  // example by the collection being saved) fails with IoError on Windows.
  // Errors: IoError.
  [[nodiscard]] Status save(const std::filesystem::path& file) const;

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
  // A failure after validation (index error) returns the error; rows before it stay inserted and
  // the failing row leaves no trace. std::bad_alloc propagates with the same semantics. Other
  // writers wait for the whole batch; searches may observe a partially inserted batch. With a
  // `pool`, normalisation runs in parallel, and so does HNSW linking in Concurrent mode. In that
  // mode a row whose linking fails with std::bad_alloc is tombstoned (its id keeps its previous
  // state) before the exception propagates.
  [[nodiscard]] Result<std::size_t> add_batch(std::span<const ExternalId> ids,
                                              std::span<const float> rows,
                                              InsertOptions options = {},
                                              ThreadPool* pool = nullptr);

  // Tombstones a vector. Errors: NotFound.
  [[nodiscard]] Status remove(ExternalId id);

  // Rebuilds the collection without removed vectors, reclaiming their memory. External ids and
  // stored vectors are kept; internal order is preserved, but an HNSW graph is rebuilt, so
  // approximate results can differ afterwards. A memory-mapped collection moves to heap memory.
  // Needs memory for both copies while it runs. Errors: ResourceExhausted; std::bad_alloc
  // propagates (the collection is unchanged on any failure).
  [[nodiscard]] Result<CompactStats> compact();

  // Copy of the stored vector (normalised for normalised collections). Errors: NotFound.
  [[nodiscard]] Result<std::vector<float>> get(ExternalId id) const;
  [[nodiscard]] bool contains(ExternalId id) const;

  // k nearest neighbours of `query`, ascending by distance (ties: older insertions first). Exact
  // for Flat; approximate for HNSW, which explores a beam of max(ef_search, k) candidates
  // (params.ef_search, else config().hnsw.ef_search) and may return fewer than k results when
  // many vectors are removed or unreachable.
  // Errors: DimensionMismatch; InvalidArgument (params, non-finite query, zero query in a
  // normalised collection).
  [[nodiscard]] Result<std::vector<Neighbor>> search(std::span<const float> query,
                                                     const SearchParams& params = {}) const;

  // As search(), writing into caller storage: requires out.size() >= params.k and returns the
  // number of results written. Flat: performs no heap allocation. HNSW: reuses pooled search
  // contexts and allocates only while they grow (first queries, a larger graph or beam width, or
  // more concurrent searches than before); may throw std::bad_alloc then.
  [[nodiscard]] Result<std::size_t> search_into(std::span<const float> query,
                                                const SearchParams& params,
                                                std::span<Neighbor> out) const;

  // Searches nq row-major queries. For query i, slots [i*k, (i+1)*k) of out_ids/out_distances
  // receive the results; counts[i] is the number of real results and unused slots are padded with
  // kInvalidExternalId / +infinity. All queries are validated before any output is written. With a
  // `pool`, queries run in parallel on its workers and the calling thread (results are identical).
  [[nodiscard]] Status search_batch(std::span<const float> queries, std::size_t nq,
                                    const SearchParams& params, std::span<ExternalId> out_ids,
                                    std::span<float> out_distances, std::span<std::uint32_t> counts,
                                    ThreadPool* pool = nullptr) const;

  // Number of searchable vectors.
  [[nodiscard]] std::size_t size() const;
  [[nodiscard]] CollectionStats stats() const;
  [[nodiscard]] const CollectionConfig& config() const noexcept;

 private:
  friend struct detail::CollectionFactory;
  struct Impl;
  explicit Collection(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vf
