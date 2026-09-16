#pragma once

// Collection and index configuration with validation.

#include <cstdint>
#include <optional>

#include <vectorforge/status.hpp>
#include <vectorforge/types.hpp>

namespace vf {

// HNSW construction and search parameters (docs/DESIGN.md §9).
struct HnswParams {
  static constexpr std::uint32_t kMinM = 2;
  static constexpr std::uint32_t kMaxM = 512;
  static constexpr std::uint32_t kMaxEf = 1U << 20;
  static constexpr std::uint8_t kMaxLevelCap = 32;

  // Links per node on levels >= 1; level 0 allows 2*M.
  std::uint32_t M = 16;  // NOLINT(readability-identifier-naming): name from the HNSW paper
  // Beam width while inserting. Must be >= M.
  std::uint32_t ef_construction = 200;
  // Default beam width for queries (per-query override allowed). Effective beam is max(ef, k).
  std::uint32_t ef_search = 50;
  // Hard cap on generated node levels.
  std::uint8_t max_level = 16;
  // Seed for deterministic level assignment.
  std::uint64_t seed = 0x5EEDF0A6EULL;

  [[nodiscard]] Status validate() const;

  [[nodiscard]] std::uint32_t max_links_level0() const noexcept { return 2 * M; }
  [[nodiscard]] std::uint32_t max_links_upper() const noexcept { return M; }
  // mL = 1 / ln(M). Precondition: validate().ok().
  [[nodiscard]] double level_multiplier() const noexcept;
};

struct CollectionConfig {
  std::uint32_t dim = 0;
  Metric metric = Metric::L2;
  // Normalise vectors to unit length on write and query. Always in effect for Metric::Cosine.
  bool normalize = false;
  IndexType index = IndexType::Hnsw;
  HnswParams hnsw{};
  // Insert/search synchronisation for HNSW (ignored by Flat). A runtime setting: not stored in
  // index files (LoadOptions::concurrency applies on load).
  Concurrency concurrency = Concurrency::Concurrent;

  [[nodiscard]] Status validate() const;

  [[nodiscard]] bool effective_normalize() const noexcept {
    return normalize || metric == Metric::Cosine;
  }
};

struct SearchParams {
  static constexpr std::uint32_t kMaxK = 1U << 20;

  // Number of neighbours requested, in [1, kMaxK]. Fewer are returned if the collection is smaller.
  std::uint32_t k = 10;
  // HNSW beam width override (ignored by Flat collections); must be in [1, HnswParams::kMaxEf].
  std::optional<std::uint32_t> ef_search;

  [[nodiscard]] Status validate() const;
};

struct InsertOptions {
  // Replace the vector of an existing id instead of failing with AlreadyExists.
  bool upsert = false;
};

// Checksum verification when loading an index file (docs/storage-format.md). Structural
// validation (bounds, sizes, graph invariants) always runs, whatever the level.
enum class Verify : std::uint8_t {
  Auto,      // Full for heap loads, Metadata for memory-mapped loads
  None,      // header and section table checksums only
  Metadata,  // every section except VECTORS (keeps mmap loads lazy)
  Full,      // every section
};

struct LoadOptions {
  // Serve vectors from a read-only memory mapping of the file instead of copying them to the heap.
  bool use_mmap = true;
  Verify verify = Verify::Auto;
  // Ask the OS to page the mapped vectors in up front (predictable first-query latency).
  bool prefault = false;
  // CollectionConfig::concurrency of the loaded collection.
  Concurrency concurrency = Concurrency::Concurrent;
};

}  // namespace vf
