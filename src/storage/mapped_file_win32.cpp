// Win32 implementation of MappedFile.

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <limits>
#include <string>
#include <system_error>
#include <utility>
#include <windows.h>

#include "storage/mapped_file.hpp"

namespace vf::detail {

namespace {

constexpr std::intptr_t kNone = -1;

Status last_error(const std::string& what, const std::filesystem::path& path) {
  const DWORD code = GetLastError();
  return Status::io_error(what + " " + path.string() + ": " +
                          std::system_category().message(static_cast<int>(code)));
}

HANDLE as_handle(std::intptr_t h) noexcept {
  return reinterpret_cast<HANDLE>(h);  // NOLINT(performance-no-int-to-ptr): opaque OS handle
}

}  // namespace

Result<MappedFile> MappedFile::open(const std::filesystem::path& path) {
  MappedFile mf;
  mf.path_ = path;
  // FILE_SHARE_DELETE lets other generations be renamed or deleted while this one is mapped.
  HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE,
                            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE) {
    return last_error("cannot open", path);
  }
  mf.file_ = reinterpret_cast<std::intptr_t>(file);
  LARGE_INTEGER size{};
  if (GetFileSizeEx(file, &size) == 0) {
    return last_error("cannot stat", path);
  }
  if (static_cast<std::uint64_t>(size.QuadPart) > std::numeric_limits<std::size_t>::max()) {
    return Status::io_error("file too large to map: " + path.string());
  }
  mf.size_ = static_cast<std::size_t>(size.QuadPart);
  if (mf.size_ == 0) {
    return mf;  // zero-length files cannot be mapped; expose an empty view
  }
  HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (mapping == nullptr) {
    return last_error("cannot create mapping for", path);
  }
  mf.mapping_ = reinterpret_cast<std::intptr_t>(mapping);
  void* view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
  if (view == nullptr) {
    return last_error("cannot map", path);
  }
  mf.data_ = static_cast<const std::byte*>(view);
  return mf;
}

MappedFile::MappedFile(MappedFile&& other) noexcept
    : path_(std::move(other.path_)),
      data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      file_(std::exchange(other.file_, kNone)),
      mapping_(std::exchange(other.mapping_, kNone)) {
}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
  if (this != &other) {
    release();
    path_ = std::move(other.path_);
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
    file_ = std::exchange(other.file_, kNone);
    mapping_ = std::exchange(other.mapping_, kNone);
  }
  return *this;
}

MappedFile::~MappedFile() {
  release();
}

void MappedFile::release() noexcept {
  if (data_ != nullptr) {
    UnmapViewOfFile(data_);
    data_ = nullptr;
  }
  if (mapping_ != kNone) {
    CloseHandle(as_handle(mapping_));
    mapping_ = kNone;
  }
  if (file_ != kNone) {
    CloseHandle(as_handle(file_));
    file_ = kNone;
  }
  size_ = 0;
}

void MappedFile::advise(AccessPattern /*pattern*/) const noexcept {
  // Windows has no per-mapping read-ahead advice; the cache manager adapts on its own.
}

void MappedFile::prefault() const noexcept {
  if (data_ == nullptr) {
    return;
  }
  WIN32_MEMORY_RANGE_ENTRY range{};
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast): the API takes a non-const pointer
  range.VirtualAddress = const_cast<void*>(static_cast<const void*>(data_));
  range.NumberOfBytes = size_;
  static_cast<void>(PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0));
}

}  // namespace vf::detail
