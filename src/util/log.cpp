#include "util/log.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <utility>

namespace vf::log {

namespace {

std::atomic<Level> g_level{Level::Info};
std::mutex g_mutex;
std::function<void(std::string_view)>& sink_slot() {
  static std::function<void(std::string_view)> sink;
  return sink;
}

bool needs_quotes(std::string_view v) noexcept {
  if (v.empty()) {
    return true;
  }
  for (const char c : v) {
    const auto u = static_cast<unsigned char>(c);
    if (u <= 0x20 || u == 0x7F || c == '"' || c == '=' || c == '\\') {
      return true;
    }
  }
  return false;
}

void append_value(std::string& out, std::string_view v) {
  if (!needs_quotes(v)) {
    out += v;
    return;
  }
  out += '"';
  for (const char c : v) {
    const auto u = static_cast<unsigned char>(c);
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (u < 0x20 || u == 0x7F) {
          std::array<char, 8> buf{};
          const int n = std::snprintf(buf.data(), buf.size(), "\\x%02x", u);
          out.append(buf.data(), static_cast<std::size_t>(n));
        } else {
          out += c;
        }
    }
  }
  out += '"';
}

void append_timestamp(std::string& out) {
  const auto now = std::chrono::system_clock::now();
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());
  const auto secs = static_cast<std::time_t>(ms.count() / 1000);
  std::tm utc{};
#if defined(_WIN32)
  gmtime_s(&utc, &secs);
#else
  gmtime_r(&secs, &utc);
#endif
  std::array<char, 32> buf{};
  const int n = std::snprintf(buf.data(), buf.size(), "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ",
                              utc.tm_year + 1900, utc.tm_mon + 1, utc.tm_mday, utc.tm_hour,
                              utc.tm_min, utc.tm_sec, static_cast<int>(ms.count() % 1000));
  out += "ts=";
  out.append(buf.data(), static_cast<std::size_t>(n));
  out += ' ';
}

}  // namespace

Field::Field(std::string_view k, double v) : key(k) {
  std::array<char, 32> buf{};
  const int n = std::snprintf(buf.data(), buf.size(), "%.6g", v);
  value.assign(buf.data(), static_cast<std::size_t>(n > 0 ? n : 0));
}

bool parse_level(std::string_view name, Level& out) noexcept {
  constexpr std::array<std::pair<std::string_view, Level>, 5> kNames{{
      {"debug", Level::Debug},
      {"info", Level::Info},
      {"warn", Level::Warn},
      {"error", Level::Error},
      {"off", Level::Off},
  }};
  for (const auto& [n, l] : kNames) {
    if (n == name) {
      out = l;
      return true;
    }
  }
  return false;
}

std::string_view to_string(Level level) noexcept {
  switch (level) {
    case Level::Debug:
      return "debug";
    case Level::Info:
      return "info";
    case Level::Warn:
      return "warn";
    case Level::Error:
      return "error";
    case Level::Off:
      return "off";
  }
  return "unknown";
}

void set_level(Level level) noexcept {
  g_level.store(level);
}

Level level() noexcept {
  return g_level.load();
}

bool enabled(Level l) noexcept {
  return l != Level::Off && l >= g_level.load();
}

void set_sink(std::function<void(std::string_view line)> sink) {
  const std::lock_guard<std::mutex> lock(g_mutex);
  sink_slot() = std::move(sink);
}

std::string format(Level level, std::string_view event, std::initializer_list<Field> fields,
                   bool timestamp) {
  std::string line;
  line.reserve(96);
  if (timestamp) {
    append_timestamp(line);
  }
  line += "level=";
  line += to_string(level);
  line += " event=";
  append_value(line, event);
  for (const Field& f : fields) {
    line += ' ';
    line += f.key;
    line += '=';
    append_value(line, f.value);
  }
  return line;
}

void write(Level l, std::string_view event, std::initializer_list<Field> fields) {
  if (!enabled(l)) {
    return;
  }
  const std::string line = format(l, event, fields, true);
  const std::lock_guard<std::mutex> lock(g_mutex);
  if (const auto& sink = sink_slot()) {
    sink(line);
    return;
  }
  std::fwrite(line.data(), 1, line.size(), stderr);
  std::fputc('\n', stderr);
  std::fflush(stderr);
}

}  // namespace vf::log
