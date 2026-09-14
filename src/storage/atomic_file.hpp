#pragma once

// Durable file output and crash-safe replacement (docs/DESIGN.md §12.4).
//
// write_atomic(target, fn): fn writes the complete contents into "<target>.tmp"; the temporary file
// is flushed and synced (fsync / FlushFileBuffers), renamed over the target (rename(2) /
// MoveFileExW with MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) and, on POSIX, the directory
// is synced. A crash at any point leaves either the previous target or the complete new one.
//
// Fault injection (tests): the four steps are "fault points". fault_injection::arm(n) makes the
// n-th fault point reached (0-based, counting across calls) return IoError immediately, without
// cleaning up, which leaves the files exactly as a process crash at that point would.

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include <vectorforge/status.hpp>

#include "storage/binary_io.hpp"

namespace vf::detail {

// Buffered, seekable output file with an explicit durability sync. Not thread-safe.
class OutputFile final : public ByteSink {
 public:
  // Creates or truncates `path`. Errors: IoError.
  [[nodiscard]] static Result<std::unique_ptr<OutputFile>> create(
      const std::filesystem::path& path);

  OutputFile(const OutputFile&) = delete;
  OutputFile& operator=(const OutputFile&) = delete;
  OutputFile(OutputFile&&) = delete;
  OutputFile& operator=(OutputFile&&) = delete;
  ~OutputFile() override;  // closes without syncing; errors are ignored

  [[nodiscard]] Status write(std::span<const std::byte> bytes) override;
  [[nodiscard]] Status write_at(std::uint64_t offset, std::span<const std::byte> bytes) override;
  // Writes buffered bytes and asks the OS to persist the file contents. Errors: IoError.
  [[nodiscard]] Status sync();
  [[nodiscard]] Status close();

 private:
  OutputFile(std::filesystem::path path, std::intptr_t handle);
  [[nodiscard]] Status flush_buffer();

  std::filesystem::path path_;
  std::intptr_t handle_;  // native handle (HANDLE or file descriptor)
  std::vector<std::byte> buffer_;
  std::uint64_t written_ = 0;  // bytes handed to the OS
};

// Atomically replaces `target` (see above). Errors: IoError, or whatever `write_contents` returns.
[[nodiscard]] Status write_atomic(const std::filesystem::path& target,
                                  const std::function<Status(ByteSink&)>& write_contents);

// Platform primitives (atomic_file_{win32,posix}.cpp).
[[nodiscard]] Status rename_replace(const std::filesystem::path& from,
                                    const std::filesystem::path& to);
// Syncs directory metadata (POSIX); no-op on Windows, where MOVEFILE_WRITE_THROUGH covers it.
[[nodiscard]] Status sync_directory(const std::filesystem::path& directory);

namespace fault_injection {

// Named fault points of write_atomic, in order.
enum class Step : std::uint8_t { TempWritten, TempSynced, Renamed, DirectorySynced };
inline constexpr std::uint64_t kStepsPerWrite = 4;

void arm(std::uint64_t fail_at_point) noexcept;
void disarm() noexcept;
[[nodiscard]] bool fired() noexcept;

// Returns IoError if this point is the armed one (and disarms), OK otherwise.
[[nodiscard]] Status check(Step step);

}  // namespace fault_injection

[[nodiscard]] std::string_view to_string(fault_injection::Step step) noexcept;

}  // namespace vf::detail
