#pragma once

// Catalog: named collections persisted under a data directory (docs/DESIGN.md §8, §12.4).
//
//   <data_dir>/collections/<name>/config.json          configuration (human-readable)
//   <data_dir>/collections/<name>/MANIFEST             current snapshot generation, if any
//   <data_dir>/collections/<name>/index.NNNNNN.vfidx   snapshot generations
//
// Names match ^[A-Za-z0-9_-]{1,64}$. A collection exists from create() on; its vectors are durable
// only up to its last snapshot(). open() restores every collection: from its MANIFEST generation,
// or empty from config.json if it was never snapshotted.
//
// Thread safety: every member function may be called concurrently. Collections are shared through
// std::shared_ptr: drop() removes a collection from the catalog at once, and its files are deleted
// when the last reference (for example an in-flight request) is released. The name can be reused
// after that. The Catalog must outlive the collections it hands out only if they are to be
// snapshotted through it; files of dropped collections are deleted even after the Catalog is gone.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <vectorforge/collection.hpp>
#include <vectorforge/config.hpp>
#include <vectorforge/status.hpp>

namespace vf {

struct SnapshotInfo {
  std::uint64_t generation = 0;
  std::uint64_t bytes = 0;  // size of the generation file
};

struct CatalogEntry {
  std::string name;
  CollectionConfig config;
  std::uint64_t size = 0;             // live vectors
  std::uint64_t last_generation = 0;  // 0: never snapshotted
};

class Catalog {
 public:
  // Opens (creating if necessary) `data_dir` and loads every collection with `options`.
  // Errors: IoError (directory not usable), CorruptData / UnsupportedVersion (a collection's files
  // are damaged; the message names the collection).
  [[nodiscard]] static Result<std::unique_ptr<Catalog>> open(const std::filesystem::path& data_dir,
                                                             const LoadOptions& options = {});

  Catalog(const Catalog&) = delete;
  Catalog& operator=(const Catalog&) = delete;
  Catalog(Catalog&&) = delete;
  Catalog& operator=(Catalog&&) = delete;
  ~Catalog();

  // Errors: InvalidArgument (name or config), AlreadyExists, Unavailable (the name belongs to a
  // dropped collection that is still in use), IoError.
  [[nodiscard]] Result<std::shared_ptr<Collection>> create(std::string_view name,
                                                           const CollectionConfig& config);
  // Errors: NotFound.
  [[nodiscard]] Result<std::shared_ptr<Collection>> get(std::string_view name) const;
  // Errors: NotFound, IoError.
  [[nodiscard]] Status drop(std::string_view name);
  // Names in ascending order.
  [[nodiscard]] std::vector<std::string> list() const;
  [[nodiscard]] std::vector<CatalogEntry> entries() const;
  [[nodiscard]] std::size_t size() const;

  // Saves a new generation of `name` (concurrent snapshots of one collection are serialised).
  // Errors: NotFound, IoError.
  [[nodiscard]] Result<SnapshotInfo> snapshot(std::string_view name);
  // Snapshots every collection; returns the first error after trying all of them.
  [[nodiscard]] Status snapshot_all();

  [[nodiscard]] const std::filesystem::path& data_dir() const noexcept;

  [[nodiscard]] static bool is_valid_name(std::string_view name) noexcept;

 private:
  struct Impl;
  explicit Catalog(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vf
