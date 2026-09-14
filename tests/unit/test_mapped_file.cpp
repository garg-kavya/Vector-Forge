// MappedFile, OutputFile and write_atomic on the real file system.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "storage/atomic_file.hpp"
#include "storage/mapped_file.hpp"
#include "support/test_data.hpp"

namespace {

using vf::ErrorCode;
using vf::detail::MappedFile;
using vf::detail::OutputFile;

void write_file(const std::filesystem::path& path, const std::string& content) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(content.data(), static_cast<std::streamsize>(content.size()));
}

std::string read_file(const std::filesystem::path& path) {
  std::string content(static_cast<std::size_t>(std::filesystem::file_size(path)), '\0');
  std::ifstream in(path, std::ios::binary);
  in.read(content.data(), static_cast<std::streamsize>(content.size()));
  return content;
}

std::span<const std::byte> bytes_of(const std::string& s) {
  return std::as_bytes(std::span<const char>(s.data(), s.size()));
}

TEST(MappedFile, MapsWholeFileContents) {
  const vf::test::ScopedTempDir dir("mmap");
  std::string content(10000, '\0');
  for (std::size_t i = 0; i < content.size(); ++i) {
    content[i] = static_cast<char>(i * 31U);
  }
  write_file(dir.file("a.bin"), content);
  vf::Result<MappedFile> mapped = MappedFile::open(dir.file("a.bin"));
  ASSERT_TRUE(mapped.ok()) << mapped.status().to_string();
  const MappedFile& mf = mapped.value();
  ASSERT_EQ(mf.size(), content.size());
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(mf.data().data()) % 4096, 0U)
      << "views start on a page boundary";
  EXPECT_TRUE(std::equal(mf.data().begin(), mf.data().end(), bytes_of(content).begin()));
  mf.advise(vf::detail::AccessPattern::Random);
  mf.advise(vf::detail::AccessPattern::Sequential);
  mf.prefault();
  EXPECT_EQ(mf.path(), dir.file("a.bin"));
}

TEST(MappedFile, EmptyMissingAndMove) {
  const vf::test::ScopedTempDir dir("mmap_edge");
  write_file(dir.file("empty.bin"), "");
  vf::Result<MappedFile> empty = MappedFile::open(dir.file("empty.bin"));
  ASSERT_TRUE(empty.ok()) << empty.status().to_string();
  EXPECT_EQ(empty.value().size(), 0U);
  EXPECT_TRUE(empty.value().data().empty());
  empty.value().prefault();

  EXPECT_EQ(MappedFile::open(dir.file("missing.bin")).status().code(), ErrorCode::IoError);

  write_file(dir.file("b.bin"), "hello mapping");
  MappedFile a = std::move(MappedFile::open(dir.file("b.bin"))).value();
  const std::byte* first = a.data().data();
  MappedFile b = std::move(a);
  EXPECT_EQ(b.data().data(), first) << "moving keeps the view";
  EXPECT_TRUE(a.data().empty());  // NOLINT(bugprone-use-after-move): moved-from state is specified
  MappedFile c;
  c = std::move(b);
  EXPECT_EQ(c.size(), 13U);
  EXPECT_EQ(static_cast<char>(c.data()[0]), 'h');
}

TEST(OutputFile, BufferedWritesPatchAndSync) {
  const vf::test::ScopedTempDir dir("output");
  const std::filesystem::path path = dir.file("out.bin");
  {
    auto file = OutputFile::create(path);
    ASSERT_TRUE(file.ok()) << file.status().to_string();
    OutputFile& out = *file.value();
    ASSERT_TRUE(out.write(bytes_of("0123456789")).ok());
    const std::string big(3 << 20, 'x');  // larger than the internal buffer
    ASSERT_TRUE(out.write(bytes_of(big)).ok());
    ASSERT_TRUE(out.write(bytes_of("tail")).ok());
    ASSERT_TRUE(out.write_at(2, bytes_of("AB")).ok());
    EXPECT_EQ(out.write_at((3 << 20) + 14, bytes_of("zz")).code(), ErrorCode::IoError)
        << "cannot patch beyond written bytes";
    ASSERT_TRUE(out.sync().ok());
    ASSERT_TRUE(out.close().ok());
    EXPECT_TRUE(out.close().ok()) << "closing twice is harmless";
    EXPECT_EQ(out.write(bytes_of("late")).code(), ErrorCode::IoError);
  }
  const std::string content = read_file(path);
  ASSERT_EQ(content.size(), 10U + (3 << 20) + 4);
  EXPECT_EQ(content.substr(0, 10), "01AB456789");
  EXPECT_EQ(content.substr(content.size() - 4), "tail");
  EXPECT_EQ(OutputFile::create(dir.file("no_such_dir") / "x.bin").status().code(),
            ErrorCode::IoError);
}

TEST(WriteAtomic, ReplacesTargetAndCleansUpOnError) {
  const vf::test::ScopedTempDir dir("atomic");
  const std::filesystem::path target = dir.file("data.bin");
  write_file(target, "old contents");
  ASSERT_TRUE(vf::detail::write_atomic(target, [](vf::detail::ByteSink& sink) {
                return sink.write(bytes_of("new contents"));
              }).ok());
  EXPECT_EQ(read_file(target), "new contents");
  EXPECT_FALSE(std::filesystem::exists(dir.file("data.bin.tmp")));

  const vf::Status failed = vf::detail::write_atomic(target, [](vf::detail::ByteSink& sink) {
    static_cast<void>(sink.write(bytes_of("partial")));
    return vf::Status::io_error("disk full (simulated)");
  });
  EXPECT_EQ(failed.code(), ErrorCode::IoError);
  EXPECT_EQ(read_file(target), "new contents") << "a failed write leaves the target untouched";
  EXPECT_FALSE(std::filesystem::exists(dir.file("data.bin.tmp"))) << "temporary file removed";

  // A new target in a fresh location works too.
  ASSERT_TRUE(vf::detail::write_atomic(dir.file("fresh.bin"), [](vf::detail::ByteSink& sink) {
                return sink.write(bytes_of("x"));
              }).ok());
  EXPECT_EQ(read_file(dir.file("fresh.bin")), "x");
}

TEST(WriteAtomic, InjectedCrashAtEveryStep) {
  const vf::test::ScopedTempDir dir("atomic_fault");
  const std::filesystem::path target = dir.file("data.bin");
  for (std::uint64_t point = 0; point < vf::detail::fault_injection::kStepsPerWrite; ++point) {
    SCOPED_TRACE(testing::Message() << "fault point " << point);
    write_file(target, "old");
    std::filesystem::remove(dir.file("data.bin.tmp"));
    vf::detail::fault_injection::arm(point);
    const vf::Status st = vf::detail::write_atomic(
        target, [](vf::detail::ByteSink& sink) { return sink.write(bytes_of("new")); });
    vf::detail::fault_injection::disarm();
    EXPECT_TRUE(vf::detail::fault_injection::fired());
    EXPECT_EQ(st.code(), ErrorCode::IoError);
    const std::string content = read_file(target);
    // Before the rename the old file is intact; from the rename on the new file is in place.
    EXPECT_EQ(content, point < 2 ? "old" : "new");
  }
  vf::detail::fault_injection::arm(100);
  EXPECT_TRUE(vf::detail::write_atomic(target, [](vf::detail::ByteSink& sink) {
                return sink.write(bytes_of("done"));
              }).ok());
  vf::detail::fault_injection::disarm();
  EXPECT_FALSE(vf::detail::fault_injection::fired()) << "a later fault point was not reached";
}

}  // namespace
