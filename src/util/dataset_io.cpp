#include "util/dataset_io.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include "core/checked_math.hpp"

namespace vf::detail {

static_assert(std::endian::native == std::endian::little,
              "VectorForge supports little-endian hosts only");

namespace {

constexpr std::string_view kNpyMagic = "\x93NUMPY";
constexpr std::size_t kMaxNpyHeaderLen = 1U << 20U;  // generous; real headers are ~100 bytes
constexpr std::uint64_t kMaxTexmexDim = 1U << 24U;

std::string path_str(const std::filesystem::path& p) {
  return p.string();
}

Status corrupt(const std::string& what) {
  return Status::corrupt_data(what);
}

std::string_view trim(std::string_view s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front())) != 0) {
    s.remove_prefix(1);
  }
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())) != 0) {
    s.remove_suffix(1);
  }
  return s;
}

// Returns the raw text following `'key':` up to (not including) the value terminator.
Result<std::string_view> dict_value(std::string_view dict, std::string_view key) {
  const std::string quoted = "'" + std::string(key) + "'";
  const std::size_t pos = dict.find(quoted);
  if (pos == std::string_view::npos) {
    return corrupt("npy header lacks key " + quoted);
  }
  std::string_view rest = dict.substr(pos + quoted.size());
  rest = trim(rest);
  if (rest.empty() || rest.front() != ':') {
    return corrupt("npy header: malformed entry for " + quoted);
  }
  rest = trim(rest.substr(1));
  return rest;
}

Result<NpyDtype> parse_descr(std::string_view value) {
  if (value.size() < 2 || (value.front() != '\'' && value.front() != '"')) {
    return corrupt("npy header: descr is not a string");
  }
  const char quote = value.front();
  const std::size_t end = value.find(quote, 1);
  if (end == std::string_view::npos) {
    return corrupt("npy header: unterminated descr");
  }
  const std::string_view descr = value.substr(1, end - 1);
  if (descr == "<f4") {
    return NpyDtype::Float32;
  }
  if (descr == "<i4") {
    return NpyDtype::Int32;
  }
  if (descr == "<i8") {
    return NpyDtype::Int64;
  }
  if (descr == "<u8") {
    return NpyDtype::UInt64;
  }
  return Status::unsupported_version("npy dtype '" + std::string(descr) +
                                     "' is not supported (expected <f4, <i4, <i8 or <u8)");
}

Result<std::vector<std::uint64_t>> parse_shape(std::string_view value) {
  if (value.empty() || value.front() != '(') {
    return corrupt("npy header: shape is not a tuple");
  }
  const std::size_t close = value.find(')');
  if (close == std::string_view::npos) {
    return corrupt("npy header: unterminated shape tuple");
  }
  std::string_view inner = value.substr(1, close - 1);
  std::vector<std::uint64_t> shape;
  while (true) {
    inner = trim(inner);
    if (inner.empty()) {
      break;
    }
    std::uint64_t dim = 0;
    const auto [ptr, ec] = std::from_chars(inner.data(), inner.data() + inner.size(), dim);
    if (ec != std::errc{}) {
      return corrupt("npy header: invalid shape entry");
    }
    shape.push_back(dim);
    inner.remove_prefix(static_cast<std::size_t>(ptr - inner.data()));
    inner = trim(inner);
    if (!inner.empty()) {
      if (inner.front() != ',') {
        return corrupt("npy header: invalid shape separator");
      }
      inner.remove_prefix(1);
    }
    if (shape.size() > 2) {
      return Status::unsupported_version(
          "npy arrays with more than 2 dimensions are not supported");
    }
  }
  if (shape.empty()) {
    return Status::unsupported_version("npy scalars (0-d arrays) are not supported");
  }
  return shape;
}

std::uint32_t read_u32_le(const char* p) noexcept {
  std::uint32_t v = 0;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

Result<std::uint64_t> file_size_of(const std::filesystem::path& path) {
  std::error_code ec;
  const std::uintmax_t size = std::filesystem::file_size(path, ec);
  if (ec) {
    return Status::io_error("cannot stat " + path_str(path) + ": " + ec.message());
  }
  return static_cast<std::uint64_t>(size);
}

template <class T>
Result<Matrix<T>> read_texmex(const std::filesystem::path& path, std::uint64_t max_rows) {
  const Result<std::uint64_t> size_or = file_size_of(path);
  if (!size_or.ok()) {
    return size_or.status();
  }
  const std::uint64_t file_size = size_or.value();
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return Status::io_error("cannot open " + path_str(path));
  }
  Matrix<T> m;
  if (file_size == 0) {
    return m;
  }
  std::array<char, 4> dim_bytes{};
  if (!in.read(dim_bytes.data(), 4)) {
    return corrupt(path_str(path) + ": truncated record header");
  }
  const auto dim_signed = static_cast<std::int32_t>(read_u32_le(dim_bytes.data()));
  if (dim_signed <= 0 || static_cast<std::uint64_t>(dim_signed) > kMaxTexmexDim) {
    return corrupt(path_str(path) + ": invalid dimension " + std::to_string(dim_signed));
  }
  const auto dim = static_cast<std::uint64_t>(dim_signed);
  const std::uint64_t record = 4 + (dim * sizeof(T));
  if (file_size % record != 0) {
    return corrupt(path_str(path) + ": file size " + std::to_string(file_size) +
                   " is not a multiple of the record size " + std::to_string(record));
  }
  std::uint64_t rows = file_size / record;
  if (max_rows != 0) {
    rows = std::min(rows, max_rows);
  }
  const auto total = checked_mul(rows, dim);
  if (!total || *total > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    return corrupt(path_str(path) + ": matrix too large");
  }
  m.rows = rows;
  m.cols = dim;
  m.data.resize(static_cast<std::size_t>(*total));
  in.seekg(0);
  for (std::uint64_t r = 0; r < rows; ++r) {
    if (!in.read(dim_bytes.data(), 4)) {
      return corrupt(path_str(path) + ": truncated at row " + std::to_string(r));
    }
    if (read_u32_le(dim_bytes.data()) != static_cast<std::uint32_t>(dim_signed)) {
      return corrupt(path_str(path) + ": row " + std::to_string(r) + " has a different dimension");
    }
    auto* const dst = reinterpret_cast<char*>(m.data.data() + (r * dim));
    if (!in.read(dst, static_cast<std::streamsize>(dim * sizeof(T)))) {
      return corrupt(path_str(path) + ": truncated at row " + std::to_string(r));
    }
  }
  return m;
}

template <class T>
Status write_texmex(const std::filesystem::path& path, std::span<const T> values,
                    std::uint64_t rows, std::uint64_t cols) {
  const auto total = checked_mul(rows, cols);
  if (!total || *total != values.size() || cols == 0 || cols > kMaxTexmexDim) {
    return Status::invalid_argument("write_texmex: values do not match rows x cols");
  }
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    return Status::io_error("cannot create " + path_str(path));
  }
  const auto dim = static_cast<std::int32_t>(cols);
  for (std::uint64_t r = 0; r < rows; ++r) {
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(values.data() + (r * cols)),
              static_cast<std::streamsize>(cols * sizeof(T)));
  }
  out.flush();
  if (!out) {
    return Status::io_error("failed writing " + path_str(path));
  }
  return {};
}

}  // namespace

const char* npy_descr(NpyDtype dtype) noexcept {
  switch (dtype) {
    case NpyDtype::Float32:
      return "<f4";
    case NpyDtype::Int32:
      return "<i4";
    case NpyDtype::Int64:
      return "<i8";
    case NpyDtype::UInt64:
      return "<u8";
  }
  return "?";
}

std::size_t npy_item_size(NpyDtype dtype) noexcept {
  switch (dtype) {
    case NpyDtype::Float32:
    case NpyDtype::Int32:
      return 4;
    case NpyDtype::Int64:
    case NpyDtype::UInt64:
      return 8;
  }
  return 0;
}

Result<NpyHeader> parse_npy_header(std::span<const char> bytes) {
  if (bytes.size() < 10 || std::string_view(bytes.data(), 6) != kNpyMagic) {
    return corrupt("not a .npy file (bad magic)");
  }
  const auto major = static_cast<unsigned char>(bytes[6]);
  std::size_t prefix = 0;
  std::uint64_t header_len = 0;
  if (major == 1) {
    prefix = 10;
    header_len = static_cast<unsigned char>(bytes[8]) |
                 (static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[9])) << 8U);
  } else if (major == 2 || major == 3) {
    if (bytes.size() < 12) {
      return corrupt("truncated .npy preamble");
    }
    prefix = 12;
    header_len = read_u32_le(bytes.data() + 8);
  } else {
    return Status::unsupported_version("unsupported .npy format version " + std::to_string(major));
  }
  if (header_len > kMaxNpyHeaderLen || prefix + header_len > bytes.size()) {
    return corrupt("truncated or oversized .npy header");
  }
  const std::string_view dict(bytes.data() + prefix, static_cast<std::size_t>(header_len));

  Result<std::string_view> descr_text = dict_value(dict, "descr");
  if (!descr_text.ok()) {
    return descr_text.status();
  }
  Result<NpyDtype> dtype = parse_descr(descr_text.value());
  if (!dtype.ok()) {
    return dtype.status();
  }
  Result<std::string_view> order_text = dict_value(dict, "fortran_order");
  if (!order_text.ok()) {
    return order_text.status();
  }
  if (order_text.value().starts_with("True")) {
    return Status::unsupported_version("Fortran-order .npy arrays are not supported");
  }
  if (!order_text.value().starts_with("False")) {
    return corrupt("npy header: invalid fortran_order");
  }
  Result<std::string_view> shape_text = dict_value(dict, "shape");
  if (!shape_text.ok()) {
    return shape_text.status();
  }
  Result<std::vector<std::uint64_t>> shape = parse_shape(shape_text.value());
  if (!shape.ok()) {
    return shape.status();
  }

  NpyHeader header;
  header.dtype = dtype.value();
  header.shape = std::move(shape).value();
  header.data_offset = prefix + header_len;
  return header;
}

std::string make_npy_header(NpyDtype dtype, std::uint64_t rows, std::uint64_t cols,
                            bool one_dimensional) {
  std::string dict = std::string("{'descr': '") + npy_descr(dtype) +
                     "', 'fortran_order': False, 'shape': (" + std::to_string(rows) +
                     (one_dimensional ? "," : ", " + std::to_string(cols)) + "), }";
  constexpr std::size_t kPrefix = 10;
  const std::size_t unpadded = kPrefix + dict.size() + 1;  // +1 for the terminating newline
  const std::size_t total = (unpadded + 63U) / 64U * 64U;
  dict.append(total - unpadded, ' ');
  dict.push_back('\n');
  const auto len = static_cast<std::uint16_t>(dict.size());
  std::string out(kNpyMagic);
  out.push_back('\x01');
  out.push_back('\x00');
  out.push_back(static_cast<char>(len & 0xFFU));
  out.push_back(static_cast<char>((len >> 8U) & 0xFFU));
  out += dict;
  return out;
}

NpyReader::NpyReader(std::filesystem::path path, std::ifstream stream, NpyHeader header)
    : path_(std::move(path)), stream_(std::move(stream)), header_(std::move(header)) {
}

Result<NpyReader> NpyReader::open(const std::filesystem::path& path) {
  const Result<std::uint64_t> size_or = file_size_of(path);
  if (!size_or.ok()) {
    return size_or.status();
  }
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return Status::io_error("cannot open " + path_str(path));
  }
  // Read the fixed-size preamble to learn the header length, then the whole preamble + header.
  std::array<char, 12> preamble{};
  const auto first = static_cast<std::size_t>(std::min<std::uint64_t>(size_or.value(), 12));
  in.read(preamble.data(), static_cast<std::streamsize>(first));
  if (first < 10 || std::string_view(preamble.data(), 6) != kNpyMagic) {
    return corrupt(path_str(path) + ": not a .npy file (bad magic)");
  }
  const auto major = static_cast<unsigned char>(preamble[6]);
  std::uint64_t total_len = 0;
  if (major == 1) {
    total_len = 10 + (static_cast<std::uint64_t>(static_cast<unsigned char>(preamble[8])) |
                      (static_cast<std::uint64_t>(static_cast<unsigned char>(preamble[9])) << 8U));
  } else if ((major == 2 || major == 3) && first == 12) {
    total_len = 12 + static_cast<std::uint64_t>(read_u32_le(preamble.data() + 8));
  } else {
    return Status::unsupported_version(path_str(path) + ": unsupported .npy format version " +
                                       std::to_string(major));
  }
  if (total_len > kMaxNpyHeaderLen + 12 || total_len > size_or.value()) {
    return corrupt(path_str(path) + ": truncated or oversized .npy header");
  }
  std::string full(static_cast<std::size_t>(total_len), '\0');
  in.seekg(0);
  if (!in.read(full.data(), static_cast<std::streamsize>(full.size()))) {
    return corrupt(path_str(path) + ": truncated .npy header");
  }
  Result<NpyHeader> header = parse_npy_header(std::span<const char>(full.data(), full.size()));
  if (!header.ok()) {
    return Status(header.status().code(), path_str(path) + ": " + header.status().message());
  }
  const NpyHeader& h = header.value();
  const std::uint64_t cols = h.shape.size() == 2 ? h.shape[1] : 1;
  const auto bytes = checked_mul(h.shape[0], cols, std::uint64_t{npy_item_size(h.dtype)});
  const auto expected = bytes ? checked_add(*bytes, h.data_offset) : std::nullopt;
  if (!expected || *expected != size_or.value()) {
    return corrupt(path_str(path) + ": file size " + std::to_string(size_or.value()) +
                   " does not match the header shape");
  }
  return NpyReader(path, std::move(in), std::move(header).value());
}

Status NpyReader::read_bytes(std::span<std::byte> out) {
  if (!stream_.read(reinterpret_cast<char*>(out.data()),
                    static_cast<std::streamsize>(out.size()))) {
    return corrupt(path_string() + ": unexpected end of data");
  }
  return {};
}

std::string NpyReader::path_string() const {
  return path_str(path_);
}

NpyWriter::NpyWriter(std::filesystem::path path, std::ofstream stream, NpyDtype dtype,
                     std::uint64_t expected)
    : path_(std::move(path)), stream_(std::move(stream)), dtype_(dtype), expected_(expected) {
}

Result<NpyWriter> NpyWriter::create(const std::filesystem::path& path, NpyDtype dtype,
                                    std::uint64_t rows, std::uint64_t cols) {
  const auto expected = checked_mul(rows, cols);
  if (!expected) {
    return Status::invalid_argument("NpyWriter: rows * cols overflows");
  }
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    return Status::io_error("cannot create " + path_str(path));
  }
  const std::string header = make_npy_header(dtype, rows, cols);
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  if (!out) {
    return Status::io_error("failed writing " + path_str(path));
  }
  return NpyWriter(path, std::move(out), dtype, *expected);
}

Status NpyWriter::write_bytes(std::span<const std::byte> bytes, std::size_t count) {
  if (written_ + count > expected_) {
    return Status::failed_precondition(path_str(path_) + ": more elements than the declared shape");
  }
  stream_.write(reinterpret_cast<const char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
  if (!stream_) {
    return Status::io_error("failed writing " + path_str(path_));
  }
  written_ += count;
  return {};
}

Status NpyWriter::finish() {
  if (written_ != expected_) {
    return Status::failed_precondition(path_str(path_) + ": wrote " + std::to_string(written_) +
                                       " of " + std::to_string(expected_) + " elements");
  }
  stream_.flush();
  stream_.close();
  if (!stream_) {
    return Status::io_error("failed closing " + path_str(path_));
  }
  return {};
}

template <class T>
Result<Matrix<T>> read_npy(const std::filesystem::path& path) {
  Result<NpyReader> reader = NpyReader::open(path);
  if (!reader.ok()) {
    return reader.status();
  }
  NpyReader& r = reader.value();
  if (r.dtype() != npy_dtype_of<T>()) {
    return Status::invalid_argument(path_str(path) + ": dtype is " + npy_descr(r.dtype()) +
                                    ", expected " + npy_descr(npy_dtype_of<T>()));
  }
  const auto total = checked_mul(r.rows(), r.cols());
  if (!total || *total > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    return corrupt(path_str(path) + ": array too large");
  }
  Matrix<T> m;
  m.rows = r.rows();
  m.cols = r.cols();
  m.data.resize(static_cast<std::size_t>(*total));
  VF_RETURN_IF_ERROR(r.read(std::span<T>(m.data)));
  return m;
}

template <class T>
Status write_npy(const std::filesystem::path& path, std::span<const T> values, std::uint64_t rows,
                 std::uint64_t cols) {
  const auto total = checked_mul(rows, cols);
  if (!total || *total != values.size()) {
    return Status::invalid_argument("write_npy: values do not match rows x cols");
  }
  Result<NpyWriter> writer = NpyWriter::create(path, npy_dtype_of<T>(), rows, cols);
  if (!writer.ok()) {
    return writer.status();
  }
  VF_RETURN_IF_ERROR(writer.value().write(values));
  return writer.value().finish();
}

template Result<Matrix<float>> read_npy<float>(const std::filesystem::path&);
template Result<Matrix<std::int32_t>> read_npy<std::int32_t>(const std::filesystem::path&);
template Result<Matrix<std::int64_t>> read_npy<std::int64_t>(const std::filesystem::path&);
template Result<Matrix<std::uint64_t>> read_npy<std::uint64_t>(const std::filesystem::path&);
template Status write_npy<float>(const std::filesystem::path&, std::span<const float>,
                                 std::uint64_t, std::uint64_t);
template Status write_npy<std::int32_t>(const std::filesystem::path&, std::span<const std::int32_t>,
                                        std::uint64_t, std::uint64_t);
template Status write_npy<std::int64_t>(const std::filesystem::path&, std::span<const std::int64_t>,
                                        std::uint64_t, std::uint64_t);
template Status write_npy<std::uint64_t>(const std::filesystem::path&,
                                         std::span<const std::uint64_t>, std::uint64_t,
                                         std::uint64_t);

Result<Matrix<float>> read_fvecs(const std::filesystem::path& path, std::uint64_t max_rows) {
  return read_texmex<float>(path, max_rows);
}

Result<Matrix<std::int32_t>> read_ivecs(const std::filesystem::path& path, std::uint64_t max_rows) {
  return read_texmex<std::int32_t>(path, max_rows);
}

Status write_fvecs(const std::filesystem::path& path, std::span<const float> values,
                   std::uint64_t rows, std::uint64_t cols) {
  return write_texmex<float>(path, values, rows, cols);
}

Status write_ivecs(const std::filesystem::path& path, std::span<const std::int32_t> values,
                   std::uint64_t rows, std::uint64_t cols) {
  return write_texmex<std::int32_t>(path, values, rows, cols);
}

Result<Matrix<float>> read_float_matrix(const std::filesystem::path& path) {
  const std::string ext = path.extension().string();
  if (ext == ".npy") {
    return read_npy<float>(path);
  }
  if (ext == ".fvecs") {
    return read_fvecs(path);
  }
  return Status::invalid_argument(path_str(path) + ": unsupported extension '" + ext +
                                  "' (expected .npy or .fvecs)");
}

}  // namespace vf::detail
