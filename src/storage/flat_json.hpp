#pragma once

// Minimal scanner for the small, flat JSON objects VectorForge writes itself (MANIFEST,
// config.json): strings without escapes, unsigned integers and booleans. Anything else is
// rejected by the callers' strict parsers.

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <system_error>

namespace vf::detail {

class FlatJsonScanner {
 public:
  explicit FlatJsonScanner(std::string_view text) noexcept : text_(text) {}

  void skip_ws() noexcept {
    while (pos_ < text_.size() && (text_[pos_] == ' ' || text_[pos_] == '\n' ||
                                   text_[pos_] == '\r' || text_[pos_] == '\t')) {
      ++pos_;
    }
  }
  bool consume(char c) noexcept {
    skip_ws();
    if (pos_ < text_.size() && text_[pos_] == c) {
      ++pos_;
      return true;
    }
    return false;
  }
  // A string without escapes.
  bool string(std::string& out) {
    if (!consume('"')) {
      return false;
    }
    const std::size_t end = text_.find('"', pos_);
    if (end == std::string_view::npos) {
      return false;
    }
    const std::string_view body = text_.substr(pos_, end - pos_);
    if (body.find('\\') != std::string_view::npos) {
      return false;
    }
    out.assign(body);
    pos_ = end + 1;
    return true;
  }
  bool number(std::uint64_t& out) noexcept {
    skip_ws();
    const char* begin = text_.data() + pos_;
    const char* end = text_.data() + text_.size();
    if (begin == end || *begin < '0' || *begin > '9') {
      return false;
    }
    const auto [ptr, ec] = std::from_chars(begin, end, out);
    if (ec != std::errc{}) {
      return false;
    }
    pos_ += static_cast<std::size_t>(ptr - begin);
    return true;
  }
  bool boolean(bool& out) noexcept {
    skip_ws();
    const std::string_view rest = text_.substr(pos_);
    if (rest.starts_with("true")) {
      out = true;
      pos_ += 4;
      return true;
    }
    if (rest.starts_with("false")) {
      out = false;
      pos_ += 5;
      return true;
    }
    return false;
  }
  [[nodiscard]] bool at_end() noexcept {
    skip_ws();
    return pos_ == text_.size();
  }

 private:
  std::string_view text_;
  std::size_t pos_ = 0;
};

}  // namespace vf::detail
