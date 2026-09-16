#include "collection/snapshot.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <optional>
#include <span>
#include <string>
#include <system_error>

#include "storage/atomic_file.hpp"
#include "storage/binary_io.hpp"
#include "storage/flat_json.hpp"
#include "storage/format.hpp"

namespace vf::detail {

namespace {

constexpr std::size_t kMaxManifestBytes = 4096;

bool is_digits(std::string_view s) noexcept {
  return !s.empty() && std::all_of(s.begin(), s.end(), [](char c) { return c >= '0' && c <= '9'; });
}

// Generation of "index.<digits>.vfidx", if the name has that form.
std::optional<std::uint64_t> parse_index_name(std::string_view name) noexcept {
  constexpr std::string_view kPrefix = "index.";
  constexpr std::string_view kSuffix = ".vfidx";
  if (!name.starts_with(kPrefix) || !name.ends_with(kSuffix)) {
    return std::nullopt;
  }
  const std::string_view digits =
      name.substr(kPrefix.size(), name.size() - kPrefix.size() - kSuffix.size());
  std::uint64_t value = 0;
  if (!is_digits(digits) ||
      std::from_chars(digits.data(), digits.data() + digits.size(), value).ec != std::errc{}) {
    return std::nullopt;
  }
  return value;
}

Result<std::uint32_t> read_header_crc(const std::filesystem::path& file) {
  std::ifstream in(file, std::ios::binary);
  std::array<char, format::kHeaderSize> header{};
  if (!in || !in.read(header.data(), header.size())) {
    return Status::io_error("cannot read header of " + file.string());
  }
  ByteReader reader(std::as_bytes(std::span<const char>(header)));
  std::uint32_t crc = 0;
  static_cast<void>(reader.seek(56));
  static_cast<void>(reader.read_u32(crc));
  return crc;
}

}  // namespace

std::string index_file_name(std::uint64_t generation) {
  std::array<char, 32> buf{};
  const int n = std::snprintf(buf.data(), buf.size(), "index.%06llu.vfidx",
                              static_cast<unsigned long long>(generation));
  return {buf.data(), static_cast<std::size_t>(n)};
}

std::string encode_manifest(const Manifest& m) {
  return R"({"format": )" + std::to_string(m.format) + R"(, "generation": )" +
         std::to_string(m.generation) + R"(, "file": ")" + m.file + R"(", "crc32c": )" +
         std::to_string(m.crc32c) + "}\n";
}

Result<Manifest> decode_manifest(std::string_view text) {
  auto bad = [](const std::string& what) { return Status::corrupt_data("MANIFEST: " + what); };
  FlatJsonScanner s(text);
  if (!s.consume('{')) {
    return bad("expected an object");
  }
  Manifest m;
  bool have_format = false;
  bool have_generation = false;
  bool have_file = false;
  bool have_crc = false;
  for (bool first = true;; first = false) {
    if (s.consume('}')) {
      if (first) {
        return bad("empty object");
      }
      break;
    }
    if (!first && !s.consume(',')) {
      return bad("expected ','");
    }
    std::string key;
    if (!s.string(key) || !s.consume(':')) {
      return bad("expected a key");
    }
    std::uint64_t number = 0;
    if (key == "format" && !have_format && s.number(number)) {
      if (number != 1) {
        return Status::unsupported_version("MANIFEST format " + std::to_string(number));
      }
      have_format = true;
    } else if (key == "generation" && !have_generation && s.number(m.generation)) {
      have_generation = true;
    } else if (key == "file" && !have_file && s.string(m.file)) {
      have_file = true;
    } else if (key == "crc32c" && !have_crc && s.number(number) && number <= 0xFFFFFFFFULL) {
      m.crc32c = static_cast<std::uint32_t>(number);
      have_crc = true;
    } else {
      return bad("unknown, duplicate or malformed key '" + key + "'");
    }
  }
  if (!s.at_end()) {
    return bad("trailing characters");
  }
  if (!have_format || !have_generation || !have_file || !have_crc) {
    return bad("missing keys");
  }
  // The file name is used to build a path: it must be exactly the generation's canonical name.
  if (parse_index_name(m.file) != m.generation || m.file != index_file_name(m.generation)) {
    return bad("file name does not match the generation");
  }
  return m;
}

Result<Manifest> read_manifest(const std::filesystem::path& directory) {
  const std::filesystem::path path = directory / kManifestName;
  std::error_code ec;
  if (!std::filesystem::exists(path, ec)) {
    return Status::not_found("no MANIFEST in " + directory.string());
  }
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return Status::io_error("cannot open " + path.string());
  }
  std::string text;
  text.resize(kMaxManifestBytes + 1);
  in.read(text.data(), static_cast<std::streamsize>(text.size()));
  if (in.bad()) {
    return Status::io_error("cannot read " + path.string());
  }
  text.resize(static_cast<std::size_t>(in.gcount()));
  if (text.size() > kMaxManifestBytes) {
    return Status::corrupt_data("MANIFEST: larger than " + std::to_string(kMaxManifestBytes) +
                                " bytes");
  }
  return decode_manifest(text);
}

Status write_manifest(const std::filesystem::path& directory, const Manifest& manifest) {
  const std::string text = encode_manifest(manifest);
  return write_atomic(directory / kManifestName, [&text](ByteSink& sink) {
    return sink.write(std::as_bytes(std::span<const char>(text.data(), text.size())));
  });
}

Result<std::uint64_t> save_snapshot(const Collection& collection,
                                    const std::filesystem::path& directory) {
  std::error_code ec;
  std::filesystem::create_directories(directory, ec);
  if (ec) {
    return Status::io_error("cannot create " + directory.string() + ": " + ec.message());
  }
  std::uint64_t generation = 1;
  Result<Manifest> current = read_manifest(directory);
  if (current.ok()) {
    generation = current.value().generation + 1;
  } else if (current.status().code() != ErrorCode::NotFound) {
    return current.status();
  }
  Manifest next;
  next.generation = generation;
  next.file = index_file_name(generation);
  const std::filesystem::path file = directory / next.file;
  VF_RETURN_IF_ERROR(collection.save(file));
  Result<std::uint32_t> crc = read_header_crc(file);
  if (!crc.ok()) {
    return crc.status();
  }
  next.crc32c = crc.value();
  VF_RETURN_IF_ERROR(write_manifest(directory, next));
  static_cast<void>(collect_garbage(directory));  // best effort; retried by the next snapshot
  return generation;
}

Result<std::unique_ptr<Collection>> load_snapshot(const std::filesystem::path& directory,
                                                  const LoadOptions& options) {
  Result<Manifest> manifest = read_manifest(directory);
  if (!manifest.ok()) {
    return manifest.status();
  }
  const std::filesystem::path file = directory / manifest.value().file;
  Result<std::uint32_t> crc = read_header_crc(file);
  if (!crc.ok()) {
    return crc.status();
  }
  if (crc.value() != manifest.value().crc32c) {
    return Status::corrupt_data("MANIFEST checksum does not match " + file.string());
  }
  return Collection::load(file, options);
}

Result<GarbageReport> collect_garbage(const std::filesystem::path& directory) {
  Result<Manifest> manifest = read_manifest(directory);
  if (!manifest.ok()) {
    return manifest.status();
  }
  GarbageReport report;
  std::error_code ec;
  std::filesystem::directory_iterator it(directory, ec);
  if (ec) {
    return Status::io_error("cannot list " + directory.string() + ": " + ec.message());
  }
  for (const std::filesystem::directory_entry& entry : it) {
    const std::string name = entry.path().filename().string();
    std::string_view base = name;
    const bool temp = base.ends_with(".tmp");
    if (temp) {
      base.remove_suffix(4);
    }
    const bool index = parse_index_name(base).has_value();
    const bool manifest_temp = temp && base == kManifestName;
    const bool doomed = (index && (temp || base != manifest.value().file)) || manifest_temp;
    if (!doomed) {
      continue;
    }
    std::error_code remove_ec;
    if (std::filesystem::remove(entry.path(), remove_ec)) {
      ++report.removed;
    } else if (remove_ec) {
      ++report.failed;
    }
  }
  return report;
}

}  // namespace vf::detail
