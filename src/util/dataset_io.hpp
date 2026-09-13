#pragma once

// Dataset file I/O: NumPy .npy (format 1.0/2.0/3.0, little-endian, C order) and TEXMEX
// .fvecs/.ivecs. Files are treated as untrusted input: headers and sizes are validated before any
// bulk read, with overflow-checked size arithmetic.
//
// VectorForge supports little-endian hosts only (docs/DESIGN.md §12.1).

#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include <vectorforge/status.hpp>

namespace vf::detail {

enum class NpyDtype : std::uint8_t { Float32, Int32, Int64, UInt64 };

// NumPy descr string ("<f4", "<i4", "<i8", "<u8").
[[nodiscard]] const char* npy_descr(NpyDtype dtype) noexcept;
[[nodiscard]] std::size_t npy_item_size(NpyDtype dtype) noexcept;

template <class T>
[[nodiscard]] constexpr NpyDtype npy_dtype_of() noexcept;
template <>
constexpr NpyDtype npy_dtype_of<float>() noexcept {
  return NpyDtype::Float32;
}
template <>
constexpr NpyDtype npy_dtype_of<std::int32_t>() noexcept {
  return NpyDtype::Int32;
}
template <>
constexpr NpyDtype npy_dtype_of<std::int64_t>() noexcept {
  return NpyDtype::Int64;
}
template <>
constexpr NpyDtype npy_dtype_of<std::uint64_t>() noexcept {
  return NpyDtype::UInt64;
}

// A dense row-major matrix. 1-D arrays load as cols == 1.
template <class T>
struct Matrix {
  std::vector<T> data;
  std::uint64_t rows = 0;
  std::uint64_t cols = 0;

  [[nodiscard]] std::span<const T> row(std::uint64_t r) const noexcept {
    return std::span<const T>(data).subspan(static_cast<std::size_t>(r * cols),
                                            static_cast<std::size_t>(cols));
  }
};

struct NpyHeader {
  NpyDtype dtype = NpyDtype::Float32;
  std::vector<std::uint64_t> shape;  // 1 or 2 dimensions
  std::uint64_t data_offset = 0;     // bytes from file start to the first element
};

// Parses a complete .npy preamble ("\x93NUMPY" magic, version, header length, header dict).
// `bytes` must contain at least the whole preamble. Errors: CorruptData, UnsupportedVersion.
[[nodiscard]] Result<NpyHeader> parse_npy_header(std::span<const char> bytes);

// Builds a format 1.0 preamble whose total length is a multiple of 64 bytes (as NumPy does).
[[nodiscard]] std::string make_npy_header(NpyDtype dtype, std::uint64_t rows, std::uint64_t cols,
                                          bool one_dimensional = false);

// Streaming .npy reader for 1-D or 2-D arrays.
class NpyReader {
 public:
  // Errors: IoError (cannot open), CorruptData (bad header or file size), UnsupportedVersion.
  [[nodiscard]] static Result<NpyReader> open(const std::filesystem::path& path);

  [[nodiscard]] NpyDtype dtype() const noexcept { return header_.dtype; }
  [[nodiscard]] std::uint64_t rows() const noexcept { return header_.shape[0]; }
  [[nodiscard]] std::uint64_t cols() const noexcept {
    return header_.shape.size() == 2 ? header_.shape[1] : 1;
  }

  // Reads the next out.size() elements. Errors: InvalidArgument (T does not match dtype),
  // CorruptData (short read).
  template <class T>
  [[nodiscard]] Status read(std::span<T> out) {
    if (npy_dtype_of<T>() != header_.dtype) {
      return Status::invalid_argument(path_string() + ": dtype is " + npy_descr(header_.dtype));
    }
    return read_bytes(std::as_writable_bytes(out));
  }

 private:
  NpyReader(std::filesystem::path path, std::ifstream stream, NpyHeader header);
  [[nodiscard]] Status read_bytes(std::span<std::byte> out);
  [[nodiscard]] std::string path_string() const;

  std::filesystem::path path_;
  std::ifstream stream_;
  NpyHeader header_;
};

// Streaming .npy writer; the shape is fixed up front and verified by finish().
class NpyWriter {
 public:
  [[nodiscard]] static Result<NpyWriter> create(const std::filesystem::path& path, NpyDtype dtype,
                                                std::uint64_t rows, std::uint64_t cols);

  template <class T>
  [[nodiscard]] Status write(std::span<const T> values) {
    if (npy_dtype_of<T>() != dtype_) {
      return Status::invalid_argument("NpyWriter: element type does not match dtype");
    }
    return write_bytes(std::as_bytes(values), values.size());
  }

  // Flushes and closes. Errors: FailedPrecondition if fewer/more elements than rows*cols were
  // written, IoError on write failure.
  [[nodiscard]] Status finish();

 private:
  NpyWriter(std::filesystem::path path, std::ofstream stream, NpyDtype dtype,
            std::uint64_t expected);
  [[nodiscard]] Status write_bytes(std::span<const std::byte> bytes, std::size_t count);

  std::filesystem::path path_;
  std::ofstream stream_;
  NpyDtype dtype_;
  std::uint64_t expected_ = 0;
  std::uint64_t written_ = 0;
};

// Whole-file helpers.
template <class T>
[[nodiscard]] Result<Matrix<T>> read_npy(const std::filesystem::path& path);
template <class T>
[[nodiscard]] Status write_npy(const std::filesystem::path& path, std::span<const T> values,
                               std::uint64_t rows, std::uint64_t cols);

// TEXMEX formats: each record is an int32 dimension followed by that many float32/int32 values.
// All records must share one dimension. `max_rows` = 0 reads all rows.
[[nodiscard]] Result<Matrix<float>> read_fvecs(const std::filesystem::path& path,
                                               std::uint64_t max_rows = 0);
[[nodiscard]] Result<Matrix<std::int32_t>> read_ivecs(const std::filesystem::path& path,
                                                      std::uint64_t max_rows = 0);
[[nodiscard]] Status write_fvecs(const std::filesystem::path& path, std::span<const float> values,
                                 std::uint64_t rows, std::uint64_t cols);
[[nodiscard]] Status write_ivecs(const std::filesystem::path& path,
                                 std::span<const std::int32_t> values, std::uint64_t rows,
                                 std::uint64_t cols);

// Loads a float32 matrix from .npy or .fvecs, chosen by file extension.
[[nodiscard]] Result<Matrix<float>> read_float_matrix(const std::filesystem::path& path);

}  // namespace vf::detail
