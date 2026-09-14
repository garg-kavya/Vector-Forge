#pragma once

// Read-only memory mapping of a whole file (docs/DESIGN.md §12.5).
//
// Win32: CreateFileW + CreateFileMappingW + MapViewOfFile of the whole file (no view offsets, so
// the 64 KB allocation-granularity rule never applies). POSIX: open + mmap(PROT_READ, MAP_SHARED).
// The view starts on a page boundary. An empty file yields an empty span and no mapping.
//
// The mapping stays valid while the object lives, even if the file is renamed. Truncating a mapped
// file is undefined (SIGBUS on POSIX); VectorForge never truncates index files in place.
// Move-only; not thread-safe to move while other threads read, but concurrent reads are fine.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>

#include <vectorforge/status.hpp>

namespace vf::detail {

enum class AccessPattern : std::uint8_t {
  Normal,
  Random,      // HNSW traversal: disables read-ahead
  Sequential,  // Flat scans: aggressive read-ahead
};

class MappedFile {
 public:
  // Errors: IoError (cannot open or map).
  [[nodiscard]] static Result<MappedFile> open(const std::filesystem::path& path);

  MappedFile() noexcept = default;
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  MappedFile(MappedFile&& other) noexcept;
  MappedFile& operator=(MappedFile&& other) noexcept;
  ~MappedFile();

  [[nodiscard]] std::span<const std::byte> data() const noexcept { return {data_, size_}; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

  // Advisory hints; failures are ignored.
  void advise(AccessPattern pattern) const noexcept;
  // Asks the OS to page the whole view in (MADV_WILLNEED / PrefetchVirtualMemory). Advisory.
  void prefault() const noexcept;

 private:
  void release() noexcept;

  std::filesystem::path path_;
  const std::byte* data_ = nullptr;
  std::size_t size_ = 0;
  std::intptr_t file_ = -1;     // Win32 file HANDLE (unused on POSIX)
  std::intptr_t mapping_ = -1;  // Win32 mapping HANDLE (unused on POSIX)
};

}  // namespace vf::detail
