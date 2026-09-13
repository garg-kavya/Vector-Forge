#pragma once

// Error handling primitives: Status and Result<T>.
//
// Expected failures (invalid input, missing data, I/O, corrupt files) are reported through these
// types rather than exceptions. Result<T> intentionally mirrors the observers of C++23
// std::expected so a later migration is mechanical. Accessing the value of a Result that holds an
// error is a programming bug and terminates the process with a diagnostic.

#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

namespace vf {

enum class ErrorCode : std::uint8_t {
  Ok = 0,
  InvalidArgument,
  DimensionMismatch,
  NotFound,
  AlreadyExists,
  CorruptData,
  UnsupportedVersion,
  IoError,
  ResourceExhausted,
  FailedPrecondition,
  Unavailable,
  Internal,
};

// Stable, upper-snake-case name used in logs and in the HTTP API ("DIMENSION_MISMATCH").
[[nodiscard]] std::string_view to_string(ErrorCode code) noexcept;

class [[nodiscard]] Status {
 public:
  // Default-constructed Status is OK.
  Status() noexcept = default;
  Status(ErrorCode code, std::string message) : code_(code), message_(std::move(message)) {}

  [[nodiscard]] static Status ok_status() noexcept { return {}; }
  [[nodiscard]] static Status invalid_argument(std::string message) {
    return {ErrorCode::InvalidArgument, std::move(message)};
  }
  [[nodiscard]] static Status dimension_mismatch(std::string message) {
    return {ErrorCode::DimensionMismatch, std::move(message)};
  }
  [[nodiscard]] static Status not_found(std::string message) {
    return {ErrorCode::NotFound, std::move(message)};
  }
  [[nodiscard]] static Status already_exists(std::string message) {
    return {ErrorCode::AlreadyExists, std::move(message)};
  }
  [[nodiscard]] static Status corrupt_data(std::string message) {
    return {ErrorCode::CorruptData, std::move(message)};
  }
  [[nodiscard]] static Status unsupported_version(std::string message) {
    return {ErrorCode::UnsupportedVersion, std::move(message)};
  }
  [[nodiscard]] static Status io_error(std::string message) {
    return {ErrorCode::IoError, std::move(message)};
  }
  [[nodiscard]] static Status resource_exhausted(std::string message) {
    return {ErrorCode::ResourceExhausted, std::move(message)};
  }
  [[nodiscard]] static Status failed_precondition(std::string message) {
    return {ErrorCode::FailedPrecondition, std::move(message)};
  }
  [[nodiscard]] static Status unavailable(std::string message) {
    return {ErrorCode::Unavailable, std::move(message)};
  }
  [[nodiscard]] static Status internal(std::string message) {
    return {ErrorCode::Internal, std::move(message)};
  }

  [[nodiscard]] bool ok() const noexcept { return code_ == ErrorCode::Ok; }
  [[nodiscard]] ErrorCode code() const noexcept { return code_; }
  [[nodiscard]] const std::string& message() const noexcept { return message_; }

  // "OK" or "CODE: message".
  [[nodiscard]] std::string to_string() const;

  friend bool operator==(const Status& lhs, const Status& rhs) noexcept {
    return lhs.code_ == rhs.code_ && lhs.message_ == rhs.message_;
  }

 private:
  ErrorCode code_ = ErrorCode::Ok;
  std::string message_;
};

namespace detail {
[[noreturn]] void result_value_on_error(const Status& status) noexcept;
[[noreturn]] void result_constructed_from_ok_status() noexcept;
}  // namespace detail

template <class T>
class [[nodiscard]] Result {
 public:
  using value_type = T;

  // NOLINTBEGIN(google-explicit-constructor): implicit conversions are the point of Result.
  Result(const T& value) : storage_(std::in_place_index<1>, value) {}
  Result(T&& value) : storage_(std::in_place_index<1>, std::move(value)) {}
  Result(Status status) : storage_(std::in_place_index<0>, std::move(status)) {
    if (std::get<0>(storage_).ok()) {
      detail::result_constructed_from_ok_status();
    }
  }
  // Converting constructor for values implicitly convertible to T (e.g. integer literals).
  template <class U>
    requires(!std::is_same_v<std::remove_cvref_t<U>, T> &&
             !std::is_same_v<std::remove_cvref_t<U>, Status> &&
             !std::is_same_v<std::remove_cvref_t<U>, Result> && std::is_convertible_v<U &&, T>)
  Result(U&& value) : storage_(std::in_place_index<1>, T(std::forward<U>(value))) {}
  // NOLINTEND(google-explicit-constructor)

  [[nodiscard]] bool has_value() const noexcept { return storage_.index() == 1; }
  [[nodiscard]] bool ok() const noexcept { return has_value(); }
  explicit operator bool() const noexcept { return has_value(); }

  [[nodiscard]] T& value() & {
    check();
    return std::get<1>(storage_);
  }
  [[nodiscard]] const T& value() const& {
    check();
    return std::get<1>(storage_);
  }
  [[nodiscard]] T&& value() && {
    check();
    return std::get<1>(std::move(storage_));
  }

  [[nodiscard]] T& operator*() & { return value(); }
  [[nodiscard]] const T& operator*() const& { return value(); }
  [[nodiscard]] T&& operator*() && { return std::move(*this).value(); }
  [[nodiscard]] T* operator->() { return &value(); }
  [[nodiscard]] const T* operator->() const { return &value(); }

  [[nodiscard]] T value_or(T fallback) const& {
    return has_value() ? std::get<1>(storage_) : std::move(fallback);
  }

  // The error. For a Result holding a value this returns an OK Status.
  [[nodiscard]] Status status() const { return has_value() ? Status{} : std::get<0>(storage_); }
  // Reference to the error without copying; an OK Status if this Result holds a value.
  [[nodiscard]] const Status& error() const& noexcept {
    static const Status ok_instance{};
    const Status* err = std::get_if<0>(&storage_);
    return err != nullptr ? *err : ok_instance;
  }

 private:
  void check() const noexcept {
    if (!has_value()) {
      detail::result_value_on_error(std::get<0>(storage_));
    }
  }

  std::variant<Status, T> storage_;
};

// Propagates a non-OK Status from the enclosing function.
#define VF_RETURN_IF_ERROR(expr)      \
  do {                                \
    ::vf::Status vf_status_ = (expr); \
    if (!vf_status_.ok()) {           \
      return vf_status_;              \
    }                                 \
  } while (false)

}  // namespace vf
