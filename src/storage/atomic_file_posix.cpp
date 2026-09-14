// POSIX implementation of native file output, rename and directory sync.

#include <cerrno>
#include <cstdio>
#include <fcntl.h>
#include <string>
#include <system_error>
#include <unistd.h>

#include "storage/atomic_file.hpp"
#include "storage/native_file.hpp"

#include <sys/stat.h>

namespace vf::detail {

namespace {

Status errno_error(const std::string& what, const std::filesystem::path& path) {
  const int code = errno;
  return Status::io_error(what + " " + path.string() + ": " +
                          std::generic_category().message(code));
}

int to_fd(native::Handle h) noexcept {
  return static_cast<int>(h);
}

}  // namespace

namespace native {

Result<Handle> create_truncate(const std::filesystem::path& path) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg): POSIX API
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (fd < 0) {
    return errno_error("cannot create", path);
  }
  return static_cast<Handle>(fd);
}

Status write_all(Handle handle, std::span<const std::byte> bytes,
                 const std::filesystem::path& path) {
  while (!bytes.empty()) {
    const ssize_t n = ::write(to_fd(handle), bytes.data(), bytes.size());
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      return errno_error("write failed for", path);
    }
    bytes = bytes.subspan(static_cast<std::size_t>(n));
  }
  return {};
}

Status write_all_at(Handle handle, std::uint64_t offset, std::span<const std::byte> bytes,
                    const std::filesystem::path& path) {
  while (!bytes.empty()) {
    const ssize_t n =
        ::pwrite(to_fd(handle), bytes.data(), bytes.size(), static_cast<off_t>(offset));
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      return errno_error("write failed for", path);
    }
    bytes = bytes.subspan(static_cast<std::size_t>(n));
    offset += static_cast<std::uint64_t>(n);
  }
  return {};
}

Status sync(Handle handle, const std::filesystem::path& path) {
  if (::fsync(to_fd(handle)) != 0) {
    return errno_error("fsync failed for", path);
  }
  return {};
}

Status close(Handle handle, const std::filesystem::path& path) {
  if (::close(to_fd(handle)) != 0) {
    return errno_error("close failed for", path);
  }
  return {};
}

}  // namespace native

Status rename_replace(const std::filesystem::path& from, const std::filesystem::path& to) {
  if (std::rename(from.c_str(), to.c_str()) != 0) {
    return errno_error("cannot rename " + from.string() + " to", to);
  }
  return {};
}

Status sync_directory(const std::filesystem::path& directory) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg): POSIX API
  const int fd = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (fd < 0) {
    return errno_error("cannot open directory", directory);
  }
  const int rc = ::fsync(fd);
  const int saved = errno;
  ::close(fd);
  if (rc != 0) {
    errno = saved;
    return errno_error("fsync failed for directory", directory);
  }
  return {};
}

}  // namespace vf::detail
