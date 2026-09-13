#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>

#include <vectorforge/status.hpp>

#include "core/assert.hpp"

namespace vf {

std::string_view to_string(ErrorCode code) noexcept {
  switch (code) {
    case ErrorCode::Ok:
      return "OK";
    case ErrorCode::InvalidArgument:
      return "INVALID_ARGUMENT";
    case ErrorCode::DimensionMismatch:
      return "DIMENSION_MISMATCH";
    case ErrorCode::NotFound:
      return "NOT_FOUND";
    case ErrorCode::AlreadyExists:
      return "ALREADY_EXISTS";
    case ErrorCode::CorruptData:
      return "CORRUPT_DATA";
    case ErrorCode::UnsupportedVersion:
      return "UNSUPPORTED_VERSION";
    case ErrorCode::IoError:
      return "IO_ERROR";
    case ErrorCode::ResourceExhausted:
      return "RESOURCE_EXHAUSTED";
    case ErrorCode::FailedPrecondition:
      return "FAILED_PRECONDITION";
    case ErrorCode::Unavailable:
      return "UNAVAILABLE";
    case ErrorCode::Internal:
      return "INTERNAL";
  }
  return "UNKNOWN";
}

std::string Status::to_string() const {
  if (ok()) {
    return "OK";
  }
  std::string out(vf::to_string(code_));
  if (!message_.empty()) {
    out += ": ";
    out += message_;
  }
  return out;
}

namespace detail {

void fatal(const char* file, int line, std::string_view message) noexcept {
  std::fprintf(stderr, "%s:%d: VectorForge fatal error: %.*s\n", file, line,
               static_cast<int>(message.size()), message.data());
  std::fflush(stderr);
  std::abort();
}

void result_value_on_error(const Status& status) noexcept {
  std::string text;
  try {
    text = "Result::value() called on an error Result: " + status.to_string();
  } catch (...) {
    text = "Result::value() called on an error Result";
  }
  fatal(__FILE__, __LINE__, text);
}

void result_constructed_from_ok_status() noexcept {
  fatal(__FILE__, __LINE__, "Result constructed from an OK Status (a Result must hold a value)");
}

}  // namespace detail
}  // namespace vf
