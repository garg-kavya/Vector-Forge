#pragma once

// Generation-based snapshots of a collection in a directory (docs/DESIGN.md §12.4,
// docs/storage-format.md "Snapshots"):
//
//   <dir>/MANIFEST              {"format":1,"generation":N,"file":"index.00000N.vfidx","crc32c":C}
//   <dir>/index.00000N.vfidx    the current generation (C = its header checksum)
//
// save_snapshot writes generation N+1 with Collection::save (atomic), then atomically replaces
// MANIFEST, then deletes other generations and leftover temporary files where possible. A crash at
// any point leaves MANIFEST naming a complete, synced generation. New generations instead of
// overwriting one file also work on Windows, where a memory-mapped file cannot be replaced.
//
// Used by the catalog (Phase 7); not thread-safe for concurrent snapshots of the same directory.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include <vectorforge/collection.hpp>
#include <vectorforge/config.hpp>
#include <vectorforge/status.hpp>

namespace vf::detail {

inline constexpr std::string_view kManifestName = "MANIFEST";

struct Manifest {
  std::uint32_t format = 1;
  std::uint64_t generation = 0;
  std::string file;          // index file name inside the directory
  std::uint32_t crc32c = 0;  // header checksum of that file

  friend bool operator==(const Manifest&, const Manifest&) = default;
};

// "index.000042.vfidx" (at least six digits).
[[nodiscard]] std::string index_file_name(std::uint64_t generation);

[[nodiscard]] std::string encode_manifest(const Manifest& manifest);
// Strict parser. Errors: CorruptData (syntax, unknown or missing keys, file name that does not
// match the generation), UnsupportedVersion (format != 1).
[[nodiscard]] Result<Manifest> decode_manifest(std::string_view text);

// Errors: NotFound (no MANIFEST), IoError, CorruptData, UnsupportedVersion.
[[nodiscard]] Result<Manifest> read_manifest(const std::filesystem::path& directory);
[[nodiscard]] Status write_manifest(const std::filesystem::path& directory,
                                    const Manifest& manifest);

// Writes the next generation and publishes it. Returns the new generation number.
// Errors: IoError, CorruptData/UnsupportedVersion (existing MANIFEST unreadable).
[[nodiscard]] Result<std::uint64_t> save_snapshot(const Collection& collection,
                                                  const std::filesystem::path& directory);

// Loads the generation named by MANIFEST after checking the file's header checksum against it.
// Errors: NotFound, IoError, CorruptData, UnsupportedVersion.
[[nodiscard]] Result<std::unique_ptr<Collection>> load_snapshot(
    const std::filesystem::path& directory, const LoadOptions& options = {});

struct GarbageReport {
  std::uint64_t removed = 0;
  std::uint64_t failed = 0;  // files still in use (e.g. mapped on Windows); retried next time
};

// Deletes index files other than the MANIFEST generation and leftover "*.tmp" files. Only names
// of the form index.<digits>.vfidx[.tmp] and MANIFEST.tmp are touched.
// Errors: NotFound/IoError/CorruptData from reading MANIFEST.
[[nodiscard]] Result<GarbageReport> collect_garbage(const std::filesystem::path& directory);

}  // namespace vf::detail
