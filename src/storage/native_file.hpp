#pragma once

// Thin wrappers over the operating system file API (atomic_file_{win32,posix}.cpp). All errors
// are reported as IoError with the path and the OS error text.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>

#include <vectorforge/status.hpp>

namespace vf::detail::native {

using Handle = std::intptr_t;
inline constexpr Handle kInvalidHandle = -1;

[[nodiscard]] Result<Handle> create_truncate(const std::filesystem::path& path);
[[nodiscard]] Status write_all(Handle handle, std::span<const std::byte> bytes,
                               const std::filesystem::path& path);
[[nodiscard]] Status write_all_at(Handle handle, std::uint64_t offset,
                                  std::span<const std::byte> bytes,
                                  const std::filesystem::path& path);
[[nodiscard]] Status sync(Handle handle, const std::filesystem::path& path);
[[nodiscard]] Status close(Handle handle, const std::filesystem::path& path);

}  // namespace vf::detail::native
