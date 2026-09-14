// Win32 implementation of native file output, rename and directory sync.

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <algorithm>
#include <limits>
#include <string>
#include <system_error>
#include <windows.h>

#include "storage/atomic_file.hpp"
#include "storage/native_file.hpp"

namespace vf::detail {

namespace {

Status last_error(const std::string& what, const std::filesystem::path& path) {
  const DWORD code = GetLastError();
  return Status::io_error(what + " " + path.string() + ": " +
                          std::system_category().message(static_cast<int>(code)));
}

HANDLE to_handle(native::Handle h) noexcept {
  return reinterpret_cast<HANDLE>(h);  // NOLINT(performance-no-int-to-ptr): opaque OS handle
}

constexpr DWORD kMaxChunk = 1U << 30U;  // WriteFile takes a 32-bit length

}  // namespace

namespace native {

Result<Handle> create_truncate(const std::filesystem::path& path) {
  HANDLE h = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                         CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h == INVALID_HANDLE_VALUE) {
    return last_error("cannot create", path);
  }
  return reinterpret_cast<Handle>(h);
}

Status write_all(Handle handle, std::span<const std::byte> bytes,
                 const std::filesystem::path& path) {
  while (!bytes.empty()) {
    const auto chunk = static_cast<DWORD>(std::min<std::size_t>(bytes.size(), kMaxChunk));
    DWORD written = 0;
    if (WriteFile(to_handle(handle), bytes.data(), chunk, &written, nullptr) == 0) {
      return last_error("write failed for", path);
    }
    bytes = bytes.subspan(written);
  }
  return {};
}

Status write_all_at(Handle handle, std::uint64_t offset, std::span<const std::byte> bytes,
                    const std::filesystem::path& path) {
  LARGE_INTEGER current{};
  LARGE_INTEGER target{};
  target.QuadPart = static_cast<LONGLONG>(offset);
  if (SetFilePointerEx(to_handle(handle), LARGE_INTEGER{}, &current, FILE_CURRENT) == 0 ||
      SetFilePointerEx(to_handle(handle), target, nullptr, FILE_BEGIN) == 0) {
    return last_error("seek failed for", path);
  }
  Status st = write_all(handle, bytes, path);
  if (SetFilePointerEx(to_handle(handle), current, nullptr, FILE_BEGIN) == 0 && st.ok()) {
    return last_error("seek failed for", path);
  }
  return st;
}

Status sync(Handle handle, const std::filesystem::path& path) {
  if (FlushFileBuffers(to_handle(handle)) == 0) {
    return last_error("flush failed for", path);
  }
  return {};
}

Status close(Handle handle, const std::filesystem::path& path) {
  if (CloseHandle(to_handle(handle)) == 0) {
    return last_error("close failed for", path);
  }
  return {};
}

}  // namespace native

Status rename_replace(const std::filesystem::path& from, const std::filesystem::path& to) {
  if (MoveFileExW(from.c_str(), to.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) ==
      0) {
    return last_error("cannot rename " + from.string() + " to", to);
  }
  return {};
}

Status sync_directory(const std::filesystem::path& /*directory*/) {
  return {};  // MOVEFILE_WRITE_THROUGH waits until the rename is flushed to disk
}

}  // namespace vf::detail
