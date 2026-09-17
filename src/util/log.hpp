#pragma once

// Minimal structured logger (docs/configuration.md, "Logging").
//
// One line per event in logfmt: `ts=2026-09-17T10:00:00.123Z level=info event=request key=value`.
// Values containing spaces, quotes or '=' are quoted with backslash escapes; control characters are
// escaped, so a line can never be split by user-controlled input. Lines go to a sink (stderr by
// default) under a mutex, one write per line.
//
// Thread safety: every function may be called concurrently.

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace vf::log {

enum class Level : std::uint8_t { Debug = 0, Info = 1, Warn = 2, Error = 3, Off = 4 };

// "debug", "info", "warn", "error", "off"; nullopt-like: returns false for unknown names.
[[nodiscard]] bool parse_level(std::string_view name, Level& out) noexcept;
[[nodiscard]] std::string_view to_string(Level level) noexcept;

struct Field {
  std::string_view key;
  std::string value;

  Field(std::string_view k, std::string_view v) : key(k), value(v) {}
  Field(std::string_view k, const char* v) : key(k), value(v) {}
  Field(std::string_view k, std::string v) : key(k), value(std::move(v)) {}
  Field(std::string_view k, bool v) : key(k), value(v ? "true" : "false") {}
  Field(std::string_view k, double v);
  template <class T>
    requires(std::is_integral_v<T> && !std::is_same_v<T, bool>)
  Field(std::string_view k, T v) : key(k), value(std::to_string(v)) {}
};

// Messages below the threshold are dropped. Default: Info.
void set_level(Level level) noexcept;
[[nodiscard]] Level level() noexcept;
[[nodiscard]] bool enabled(Level level) noexcept;

// Replaces the output (nullptr restores stderr). The sink receives complete lines without '\n'.
void set_sink(std::function<void(std::string_view line)> sink);

void write(Level level, std::string_view event, std::initializer_list<Field> fields = {});

// Formats one line (without timestamp when `timestamp` is false); exposed for tests.
[[nodiscard]] std::string format(Level level, std::string_view event,
                                 std::initializer_list<Field> fields, bool timestamp);

}  // namespace vf::log
