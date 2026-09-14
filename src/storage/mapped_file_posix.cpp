// POSIX implementation of MappedFile.

#include <cerrno>
#include <fcntl.h>
#include <limits>
#include <string>
#include <system_error>
#include <unistd.h>
#include <utility>

#include "storage/mapped_file.hpp"

#include <sys/mman.h>
#include <sys/stat.h>

namespace vf::detail {

namespace {

Status errno_error(const std::string& what, const std::filesystem::path& path) {
  const int code = errno;
  return Status::io_error(what + " " + path.string() + ": " +
                          std::generic_category().message(code));
}

}  // namespace

Result<MappedFile> MappedFile::open(const std::filesystem::path& path) {
  MappedFile mf;
  mf.path_ = path;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg): POSIX API
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    return errno_error("cannot open", path);
  }
  struct stat st{};
  if (::fstat(fd, &st) != 0) {
    Status err = errno_error("cannot stat", path);
    ::close(fd);
    return err;
  }
  if (st.st_size < 0 ||
      static_cast<std::uint64_t>(st.st_size) > std::numeric_limits<std::size_t>::max()) {
    ::close(fd);
    return Status::io_error("file too large to map: " + path.string());
  }
  mf.size_ = static_cast<std::size_t>(st.st_size);
  if (mf.size_ == 0) {
    ::close(fd);
    return mf;
  }
  void* view = ::mmap(nullptr, mf.size_, PROT_READ, MAP_SHARED, fd, 0);
  const int saved = errno;
  ::close(fd);               // the mapping keeps the file referenced
  if (view == MAP_FAILED) {  // NOLINT(cppcoreguidelines-pro-type-cstyle-cast): POSIX macro
    errno = saved;
    mf.size_ = 0;
    return errno_error("cannot map", path);
  }
  mf.data_ = static_cast<const std::byte*>(view);
  return mf;
}

MappedFile::MappedFile(MappedFile&& other) noexcept
    : path_(std::move(other.path_)),
      data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      file_(other.file_),
      mapping_(other.mapping_) {
}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
  if (this != &other) {
    release();
    path_ = std::move(other.path_);
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
  }
  return *this;
}

MappedFile::~MappedFile() {
  release();
}

void MappedFile::release() noexcept {
  if (data_ != nullptr) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast): munmap takes a non-const pointer
    ::munmap(const_cast<void*>(static_cast<const void*>(data_)), size_);
    data_ = nullptr;
  }
  size_ = 0;
}

void MappedFile::advise(AccessPattern pattern) const noexcept {
  if (data_ == nullptr) {
    return;
  }
  int advice = MADV_NORMAL;
  if (pattern == AccessPattern::Random) {
    advice = MADV_RANDOM;
  } else if (pattern == AccessPattern::Sequential) {
    advice = MADV_SEQUENTIAL;
  }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast): madvise takes a non-const pointer
  static_cast<void>(::madvise(const_cast<void*>(static_cast<const void*>(data_)), size_, advice));
}

void MappedFile::prefault() const noexcept {
  if (data_ != nullptr) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast): madvise takes a non-const pointer
    static_cast<void>(
        ::madvise(const_cast<void*>(static_cast<const void*>(data_)), size_, MADV_WILLNEED));
  }
}

}  // namespace vf::detail
