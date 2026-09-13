#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "support/test_data.hpp"
#include "util/dataset_io.hpp"

namespace {

using vf::ErrorCode;
using vf::detail::Matrix;
using vf::detail::NpyDtype;

void write_raw(const std::filesystem::path& path, const std::string& bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

std::string read_raw(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  const auto size = static_cast<std::size_t>(std::filesystem::file_size(path));
  std::string bytes(size, char{0});
  in.read(bytes.data(), static_cast<std::streamsize>(size));
  return bytes;
}

// Reproduces exactly what NumPy 2.4 writes for np.save (header padded to 128 bytes), captured from
// `np.save(io.BytesIO(), arr)` on the development machine.
std::string numpy_preamble(const std::string& dict_prefix) {
  std::string dict = dict_prefix;
  dict.append(117 - dict.size(), ' ');
  dict.push_back('\n');
  std::string out = "\x93NUMPY";
  out.push_back('\x01');
  out.push_back('\x00');
  out.push_back(static_cast<char>(dict.size()));
  out.push_back('\x00');
  return out + dict;
}

std::string bytes_of(const std::vector<float>& v) {
  return {reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float)};
}

TEST(Npy, RoundTripAllDtypes) {
  const vf::test::ScopedTempDir dir("npy_roundtrip");
  const std::vector<float> f = {1.5F, -2.0F, 3.25F, 4.0F, 5.0F, -6.5F};
  ASSERT_TRUE(vf::detail::write_npy(dir.file("f.npy"), std::span<const float>(f), 2, 3).ok());
  const auto rf = vf::detail::read_npy<float>(dir.file("f.npy"));
  ASSERT_TRUE(rf.ok()) << rf.status().to_string();
  EXPECT_EQ(rf.value().rows, 2U);
  EXPECT_EQ(rf.value().cols, 3U);
  EXPECT_EQ(rf.value().data, f);
  EXPECT_EQ(rf.value().row(1)[2], -6.5F);

  const std::vector<std::int64_t> i64 = {-1, 0, std::numeric_limits<std::int64_t>::max()};
  ASSERT_TRUE(
      vf::detail::write_npy(dir.file("i64.npy"), std::span<const std::int64_t>(i64), 3, 1).ok());
  EXPECT_EQ(vf::detail::read_npy<std::int64_t>(dir.file("i64.npy")).value().data, i64);

  const std::vector<std::uint64_t> u64 = {0, std::numeric_limits<std::uint64_t>::max()};
  ASSERT_TRUE(
      vf::detail::write_npy(dir.file("u64.npy"), std::span<const std::uint64_t>(u64), 1, 2).ok());
  EXPECT_EQ(vf::detail::read_npy<std::uint64_t>(dir.file("u64.npy")).value().data, u64);

  const std::vector<std::int32_t> i32 = {7, -7};
  ASSERT_TRUE(
      vf::detail::write_npy(dir.file("i32.npy"), std::span<const std::int32_t>(i32), 2, 1).ok());
  EXPECT_EQ(vf::detail::read_npy<std::int32_t>(dir.file("i32.npy")).value().data, i32);

  // Empty array.
  ASSERT_TRUE(vf::detail::write_npy(dir.file("empty.npy"), std::span<const float>(), 0, 5).ok());
  const auto empty = vf::detail::read_npy<float>(dir.file("empty.npy"));
  ASSERT_TRUE(empty.ok());
  EXPECT_EQ(empty.value().rows, 0U);
  EXPECT_EQ(empty.value().cols, 5U);
}

TEST(Npy, WrittenHeaderIsAlignedAndNumpyCompatible) {
  const std::string h = vf::detail::make_npy_header(NpyDtype::Float32, 3, 4);
  EXPECT_EQ(h.size() % 64, 0U);
  EXPECT_EQ(h.back(), '\n');
  EXPECT_EQ(h.substr(0, 6), "\x93NUMPY");
  EXPECT_NE(h.find("{'descr': '<f4', 'fortran_order': False, 'shape': (3, 4), }"),
            std::string::npos);
  const auto parsed = vf::detail::parse_npy_header(std::span<const char>(h.data(), h.size()));
  ASSERT_TRUE(parsed.ok());
  EXPECT_EQ(parsed.value().data_offset, h.size());

  const std::string h1 =
      vf::detail::make_npy_header(NpyDtype::Int64, 5, 1, /*one_dimensional=*/true);
  EXPECT_NE(h1.find("'shape': (5,), }"), std::string::npos);
}

TEST(Npy, ReadsNumpyWrittenFiles) {
  const vf::test::ScopedTempDir dir("npy_numpy");
  std::vector<float> values(12);
  for (int i = 0; i < 12; ++i) {
    values[static_cast<std::size_t>(i)] = static_cast<float>(i);
  }
  write_raw(dir.file("a.npy"),
            numpy_preamble("{'descr': '<f4', 'fortran_order': False, 'shape': (3, 4), }") +
                bytes_of(values));
  const auto m = vf::detail::read_npy<float>(dir.file("a.npy"));
  ASSERT_TRUE(m.ok()) << m.status().to_string();
  EXPECT_EQ(m.value().rows, 3U);
  EXPECT_EQ(m.value().cols, 4U);
  EXPECT_EQ(m.value().data, values);

  const std::vector<std::int64_t> ids = {1, -2, 3, -4, 5};
  write_raw(dir.file("b.npy"),
            numpy_preamble("{'descr': '<i8', 'fortran_order': False, 'shape': (5,), }") +
                std::string(reinterpret_cast<const char*>(ids.data()), ids.size() * 8));
  const auto b = vf::detail::read_npy<std::int64_t>(dir.file("b.npy"));
  ASSERT_TRUE(b.ok()) << b.status().to_string();
  EXPECT_EQ(b.value().rows, 5U);
  EXPECT_EQ(b.value().cols, 1U);
  EXPECT_EQ(b.value().data, ids);
}

TEST(Npy, RejectsInvalidFiles) {
  const vf::test::ScopedTempDir dir("npy_invalid");
  const std::string good_data = bytes_of(std::vector<float>(6, 1.0F));
  auto expect_code = [&](const std::string& name, const std::string& bytes, ErrorCode code) {
    write_raw(dir.file(name), bytes);
    EXPECT_EQ(vf::detail::read_npy<float>(dir.file(name)).status().code(), code) << name;
  };
  expect_code("magic.npy", "\x93NUMPZ" + std::string(200, ' '), ErrorCode::CorruptData);
  expect_code("tiny.npy", "\x93NU", ErrorCode::CorruptData);
  expect_code(
      "fortran.npy",
      numpy_preamble("{'descr': '<f4', 'fortran_order': True, 'shape': (2, 3), }") + good_data,
      ErrorCode::UnsupportedVersion);
  expect_code("f64.npy",
              numpy_preamble("{'descr': '<f8', 'fortran_order': False, 'shape': (2, 3), }") +
                  std::string(48, '\0'),
              ErrorCode::UnsupportedVersion);
  expect_code(
      "bigendian.npy",
      numpy_preamble("{'descr': '>f4', 'fortran_order': False, 'shape': (2, 3), }") + good_data,
      ErrorCode::UnsupportedVersion);
  expect_code(
      "3d.npy",
      numpy_preamble("{'descr': '<f4', 'fortran_order': False, 'shape': (1, 2, 3), }") + good_data,
      ErrorCode::UnsupportedVersion);
  expect_code("short.npy",
              numpy_preamble("{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }") +
                  good_data.substr(0, 20),
              ErrorCode::CorruptData);
  expect_code("long.npy",
              numpy_preamble("{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }") +
                  good_data + "x",
              ErrorCode::CorruptData);
  expect_code("overflow.npy",
              numpy_preamble(
                  "{'descr': '<f4', 'fortran_order': False, 'shape': (4294967296, 4294967296), }") +
                  good_data,
              ErrorCode::CorruptData);
  expect_code("noshape.npy",
              numpy_preamble("{'descr': '<f4', 'fortran_order': False, }") + good_data,
              ErrorCode::CorruptData);
  std::string truncated_header =
      numpy_preamble("{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }");
  expect_code("trunc_header.npy", truncated_header.substr(0, 40), ErrorCode::CorruptData);
  std::string v9 = truncated_header;
  v9[6] = '\x09';
  expect_code("version.npy", v9 + good_data, ErrorCode::UnsupportedVersion);

  EXPECT_EQ(vf::detail::read_npy<float>(dir.file("missing.npy")).status().code(),
            ErrorCode::IoError);

  // Correct file, wrong requested dtype.
  const std::vector<float> f = {1, 2};
  ASSERT_TRUE(vf::detail::write_npy(dir.file("f.npy"), std::span<const float>(f), 1, 2).ok());
  EXPECT_EQ(vf::detail::read_npy<std::int64_t>(dir.file("f.npy")).status().code(),
            ErrorCode::InvalidArgument);
}

TEST(Npy, WriterEnforcesShape) {
  const vf::test::ScopedTempDir dir("npy_writer");
  auto writer = vf::detail::NpyWriter::create(dir.file("w.npy"), NpyDtype::Float32, 2, 2);
  ASSERT_TRUE(writer.ok());
  const std::vector<float> row = {1.0F, 2.0F};
  ASSERT_TRUE(writer.value().write(std::span<const float>(row)).ok());
  EXPECT_EQ(writer.value().finish().code(), ErrorCode::FailedPrecondition) << "only half written";

  auto w2 = vf::detail::NpyWriter::create(dir.file("w2.npy"), NpyDtype::Float32, 1, 2);
  ASSERT_TRUE(w2.ok());
  const std::vector<float> three = {1, 2, 3};
  EXPECT_EQ(w2.value().write(std::span<const float>(three)).code(), ErrorCode::FailedPrecondition);
  const std::vector<std::int64_t> wrong = {1, 2};
  EXPECT_EQ(w2.value().write(std::span<const std::int64_t>(wrong)).code(),
            ErrorCode::InvalidArgument);

  // Streaming in pieces produces the same bytes as a one-shot write.
  const std::vector<float> all = {1, 2, 3, 4, 5, 6};
  ASSERT_TRUE(
      vf::detail::write_npy(dir.file("oneshot.npy"), std::span<const float>(all), 3, 2).ok());
  auto w3 = vf::detail::NpyWriter::create(dir.file("stream.npy"), NpyDtype::Float32, 3, 2);
  ASSERT_TRUE(w3.ok());
  for (std::size_t r = 0; r < 3; ++r) {
    ASSERT_TRUE(w3.value().write(std::span<const float>(all).subspan(r * 2, 2)).ok());
  }
  ASSERT_TRUE(w3.value().finish().ok());
  EXPECT_EQ(read_raw(dir.file("oneshot.npy")), read_raw(dir.file("stream.npy")));
}

TEST(Texmex, FvecsIvecsRoundTrip) {
  const vf::test::ScopedTempDir dir("texmex");
  const std::vector<float> f = {1, 2, 3, 4, 5, 6};
  ASSERT_TRUE(vf::detail::write_fvecs(dir.file("a.fvecs"), f, 3, 2).ok());
  const auto rf = vf::detail::read_fvecs(dir.file("a.fvecs"));
  ASSERT_TRUE(rf.ok()) << rf.status().to_string();
  EXPECT_EQ(rf.value().rows, 3U);
  EXPECT_EQ(rf.value().cols, 2U);
  EXPECT_EQ(rf.value().data, f);
  EXPECT_EQ(read_raw(dir.file("a.fvecs")).size(), 3U * (4 + 8));

  const auto limited = vf::detail::read_fvecs(dir.file("a.fvecs"), 2);
  ASSERT_TRUE(limited.ok());
  EXPECT_EQ(limited.value().rows, 2U);

  const std::vector<std::int32_t> iv = {9, 8, 7, 6, 5, 4};
  ASSERT_TRUE(vf::detail::write_ivecs(dir.file("b.ivecs"), iv, 2, 3).ok());
  EXPECT_EQ(vf::detail::read_ivecs(dir.file("b.ivecs")).value().data, iv);

  const auto via_ext = vf::detail::read_float_matrix(dir.file("a.fvecs"));
  ASSERT_TRUE(via_ext.ok());
  EXPECT_EQ(via_ext.value().data, f);
  write_raw(dir.file("c.txt"), "x");
  EXPECT_EQ(vf::detail::read_float_matrix(dir.file("c.txt")).status().code(),
            ErrorCode::InvalidArgument);

  write_raw(dir.file("empty.fvecs"), "");
  EXPECT_EQ(vf::detail::read_fvecs(dir.file("empty.fvecs")).value().rows, 0U);
}

TEST(Texmex, RejectsInvalidFiles) {
  const vf::test::ScopedTempDir dir("texmex_invalid");
  std::string good = read_raw([&] {
    const std::vector<float> f = {1, 2, 3, 4};
    EXPECT_TRUE(vf::detail::write_fvecs(dir.file("g.fvecs"), f, 2, 2).ok());
    return dir.file("g.fvecs");
  }());
  write_raw(dir.file("partial.fvecs"), good.substr(0, good.size() - 1));
  EXPECT_EQ(vf::detail::read_fvecs(dir.file("partial.fvecs")).status().code(),
            ErrorCode::CorruptData);

  std::string mixed = good;
  mixed[12] = '\x03';  // second record claims dim 3
  write_raw(dir.file("mixed.fvecs"), mixed);
  EXPECT_EQ(vf::detail::read_fvecs(dir.file("mixed.fvecs")).status().code(),
            ErrorCode::CorruptData);

  std::string negative = good;
  negative[3] = '\x80';
  write_raw(dir.file("neg.fvecs"), negative);
  EXPECT_EQ(vf::detail::read_fvecs(dir.file("neg.fvecs")).status().code(), ErrorCode::CorruptData);
}

}  // namespace
