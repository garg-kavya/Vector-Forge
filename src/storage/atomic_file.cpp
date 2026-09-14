#include "storage/atomic_file.hpp"

#include <atomic>
#include <string>
#include <system_error>
#include <utility>

#include "storage/native_file.hpp"

namespace vf::detail {

namespace {

constexpr std::size_t kBufferBytes = std::size_t{1} << 20U;  // 1 MiB

}  // namespace

Result<std::unique_ptr<OutputFile>> OutputFile::create(const std::filesystem::path& path) {
  Result<native::Handle> handle = native::create_truncate(path);
  if (!handle.ok()) {
    return handle.status();
  }
  return std::unique_ptr<OutputFile>(new OutputFile(path, handle.value()));
}

OutputFile::OutputFile(std::filesystem::path path, std::intptr_t handle)
    : path_(std::move(path)), handle_(handle) {
  buffer_.reserve(kBufferBytes);
}

OutputFile::~OutputFile() {
  if (handle_ != native::kInvalidHandle) {
    static_cast<void>(native::close(handle_, path_));
  }
}

Status OutputFile::flush_buffer() {
  if (buffer_.empty()) {
    return {};
  }
  VF_RETURN_IF_ERROR(native::write_all(handle_, buffer_, path_));
  written_ += buffer_.size();
  buffer_.clear();
  return {};
}

Status OutputFile::write(std::span<const std::byte> bytes) {
  if (handle_ == native::kInvalidHandle) {
    return Status::io_error("write to closed file " + path_.string());
  }
  if (buffer_.size() + bytes.size() <= kBufferBytes) {
    buffer_.insert(buffer_.end(), bytes.begin(), bytes.end());
    return {};
  }
  VF_RETURN_IF_ERROR(flush_buffer());
  if (bytes.size() >= kBufferBytes) {
    VF_RETURN_IF_ERROR(native::write_all(handle_, bytes, path_));
    written_ += bytes.size();
    return {};
  }
  buffer_.insert(buffer_.end(), bytes.begin(), bytes.end());
  return {};
}

Status OutputFile::write_at(std::uint64_t offset, std::span<const std::byte> bytes) {
  if (handle_ == native::kInvalidHandle) {
    return Status::io_error("write to closed file " + path_.string());
  }
  VF_RETURN_IF_ERROR(flush_buffer());
  if (offset > written_ || bytes.size() > written_ - offset) {
    return Status::io_error("write_at beyond written data in " + path_.string());
  }
  return native::write_all_at(handle_, offset, bytes, path_);
}

Status OutputFile::sync() {
  VF_RETURN_IF_ERROR(flush_buffer());
  return native::sync(handle_, path_);
}

Status OutputFile::close() {
  if (handle_ == native::kInvalidHandle) {
    return {};
  }
  Status flushed = flush_buffer();
  const native::Handle handle = std::exchange(handle_, native::kInvalidHandle);
  Status closed = native::close(handle, path_);
  return flushed.ok() ? closed : flushed;
}

Status write_atomic(const std::filesystem::path& target,
                    const std::function<Status(ByteSink&)>& write_contents) {
  std::filesystem::path temp = target;
  temp += ".tmp";
  std::filesystem::path directory = target.parent_path();
  if (directory.empty()) {
    directory = ".";
  }

  // On a genuine error the temporary file is removed; an injected fault returns immediately.
  auto remove_temp = [&temp] {
    std::error_code ec;
    std::filesystem::remove(temp, ec);
  };

  {
    Result<std::unique_ptr<OutputFile>> file = OutputFile::create(temp);
    if (!file.ok()) {
      return file.status();
    }
    OutputFile& out = *file.value();
    if (Status st = write_contents(out); !st.ok()) {
      static_cast<void>(out.close());
      remove_temp();
      return st;
    }
    VF_RETURN_IF_ERROR(fault_injection::check(fault_injection::Step::TempWritten));
    if (Status st = out.sync(); !st.ok()) {
      static_cast<void>(out.close());
      remove_temp();
      return st;
    }
    if (Status st = out.close(); !st.ok()) {
      remove_temp();
      return st;
    }
  }
  VF_RETURN_IF_ERROR(fault_injection::check(fault_injection::Step::TempSynced));
  if (Status st = rename_replace(temp, target); !st.ok()) {
    remove_temp();
    return st;
  }
  VF_RETURN_IF_ERROR(fault_injection::check(fault_injection::Step::Renamed));
  VF_RETURN_IF_ERROR(sync_directory(directory));
  return fault_injection::check(fault_injection::Step::DirectorySynced);
}

std::string_view to_string(fault_injection::Step step) noexcept {
  switch (step) {
    case fault_injection::Step::TempWritten:
      return "temp-written";
    case fault_injection::Step::TempSynced:
      return "temp-synced";
    case fault_injection::Step::Renamed:
      return "renamed";
    case fault_injection::Step::DirectorySynced:
      return "directory-synced";
  }
  return "unknown";
}

namespace fault_injection {

namespace {

constexpr std::uint64_t kDisarmed = ~std::uint64_t{0};
std::atomic<std::uint64_t> g_remaining{kDisarmed};
std::atomic<bool> g_fired{false};

}  // namespace

void arm(std::uint64_t fail_at_point) noexcept {
  g_fired.store(false);
  g_remaining.store(fail_at_point);
}

void disarm() noexcept {
  g_remaining.store(kDisarmed);
}

bool fired() noexcept {
  return g_fired.load();
}

Status check(Step step) {
  std::uint64_t remaining = g_remaining.load();
  while (remaining != kDisarmed) {
    const std::uint64_t next = remaining == 0 ? kDisarmed : remaining - 1;
    if (g_remaining.compare_exchange_weak(remaining, next)) {
      if (remaining == 0) {
        g_fired.store(true);
        return Status::io_error("injected crash after step " + std::string(to_string(step)));
      }
      return {};
    }
  }
  return {};
}

}  // namespace fault_injection

}  // namespace vf::detail
