// Crash safety of Collection::save and generation snapshots (docs/DESIGN.md §12.4). A simulated
// crash is injected at every fault point; afterwards the file or snapshot directory must load as
// either the complete old state or the complete new state, and the next save must succeed.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <vectorforge/collection.hpp>

#include "collection/snapshot.hpp"
#include "storage/atomic_file.hpp"
#include "support/persistence.hpp"
#include "support/test_data.hpp"

namespace {

using vf::Collection;
using vf::ErrorCode;
namespace fi = vf::detail::fault_injection;

constexpr std::uint32_t kDim = 6;

std::unique_ptr<Collection> make(std::size_t rows) {
  auto c =
      Collection::create(vf::test::small_config(vf::IndexType::Hnsw, vf::Metric::L2, false, kDim))
          .value();
  vf::test::apply_operations(*c, 4, rows, 0);
  return c;
}

std::vector<std::string> listing(const std::filesystem::path& dir) {
  std::vector<std::string> names;
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    names.push_back(entry.path().filename().string());
  }
  std::sort(names.begin(), names.end());
  return names;
}

void touch(const std::filesystem::path& path) {
  std::ofstream(path, std::ios::binary) << "x";
}

TEST(AtomicSave, CrashAtEveryStepLeavesOldOrNewFile) {
  const auto old_state = make(100);
  const auto new_state = make(160);
  const std::uint64_t old_rows = old_state->stats().row_count;
  const std::uint64_t new_rows = new_state->stats().row_count;
  ASSERT_NE(old_rows, new_rows);
  for (std::uint64_t point = 0; point < fi::kStepsPerWrite; ++point) {
    SCOPED_TRACE(testing::Message()
                 << "crash after step " << vf::detail::to_string(static_cast<fi::Step>(point)));
    const vf::test::ScopedTempDir dir("atomic_save");
    const std::filesystem::path file = dir.file("c.vfidx");
    ASSERT_TRUE(old_state->save(file).ok());
    fi::arm(point);
    const vf::Status st = new_state->save(file);
    fi::disarm();
    ASSERT_TRUE(fi::fired());
    EXPECT_EQ(st.code(), ErrorCode::IoError);
    const auto loaded = Collection::load(file, {.use_mmap = false, .verify = vf::Verify::Full});
    ASSERT_TRUE(loaded.ok()) << loaded.status().to_string();
    EXPECT_EQ(loaded.value()->stats().row_count, point < 2 ? old_rows : new_rows);
    // Recovery: the next save succeeds and replaces any leftover temporary file.
    ASSERT_TRUE(new_state->save(file).ok());
    EXPECT_EQ(listing(dir.path()), (std::vector<std::string>{"c.vfidx"}));
    EXPECT_EQ(Collection::load(file).value()->stats().row_count, new_rows);
  }
}

TEST(Snapshot, CrashAtEveryStepLeavesOldOrNewGeneration) {
  const auto old_state = make(80);
  const auto new_state = make(140);
  const std::uint64_t old_rows = old_state->stats().row_count;
  const std::uint64_t new_rows = new_state->stats().row_count;
  // Two atomic writes per snapshot: the index file (points 0-3) and MANIFEST (points 4-7). MANIFEST
  // is published by its rename, fault point 6.
  for (std::uint64_t point = 0; point < 2 * fi::kStepsPerWrite; ++point) {
    SCOPED_TRACE(testing::Message() << "fault point " << point);
    const vf::test::ScopedTempDir dir("snapshot_crash");
    ASSERT_EQ(vf::detail::save_snapshot(*old_state, dir.path()).value(), 1U);
    fi::arm(point);
    const vf::Result<std::uint64_t> gen = vf::detail::save_snapshot(*new_state, dir.path());
    fi::disarm();
    ASSERT_TRUE(fi::fired());
    EXPECT_EQ(gen.status().code(), ErrorCode::IoError);

    const bool published = point >= 6;
    const auto loaded = vf::detail::load_snapshot(dir.path(), {.use_mmap = false});
    ASSERT_TRUE(loaded.ok()) << loaded.status().to_string();
    EXPECT_EQ(loaded.value()->stats().row_count, published ? new_rows : old_rows);
    EXPECT_EQ(vf::detail::read_manifest(dir.path()).value().generation, published ? 2U : 1U);

    // Recovery: garbage collection plus another snapshot converge to one clean generation.
    ASSERT_TRUE(vf::detail::collect_garbage(dir.path()).ok());
    const vf::Result<std::uint64_t> next = vf::detail::save_snapshot(*new_state, dir.path());
    ASSERT_TRUE(next.ok()) << next.status().to_string();
    EXPECT_EQ(next.value(), published ? 3U : 2U);
    EXPECT_EQ(listing(dir.path()),
              (std::vector<std::string>{"MANIFEST", vf::detail::index_file_name(next.value())}));
    EXPECT_EQ(vf::detail::load_snapshot(dir.path()).value()->stats().row_count, new_rows);
  }
}

TEST(Snapshot, GenerationsAndGarbageCollection) {
  const vf::test::ScopedTempDir dir("snapshot_gc");
  EXPECT_EQ(vf::detail::load_snapshot(dir.path()).status().code(), ErrorCode::NotFound);
  EXPECT_EQ(vf::detail::collect_garbage(dir.path()).status().code(), ErrorCode::NotFound);
  const auto state = make(50);
  EXPECT_EQ(vf::detail::save_snapshot(*state, dir.path() / "nested").value(), 1U)
      << "the directory is created";
  const std::filesystem::path d = dir.path() / "nested";
  EXPECT_EQ(vf::detail::save_snapshot(*state, d).value(), 2U);
  EXPECT_EQ(vf::detail::save_snapshot(*state, d).value(), 3U);
  EXPECT_EQ(listing(d), (std::vector<std::string>{"MANIFEST", "index.000003.vfidx"}));

  // Only VectorForge's own leftovers are collected.
  touch(d / "notes.txt");
  touch(d / "index.000001.vfidx");
  touch(d / "index.000009.vfidx.tmp");
  touch(d / "MANIFEST.tmp");
  touch(d / "index.abc.vfidx");
  const vf::Result<vf::detail::GarbageReport> report = vf::detail::collect_garbage(d);
  ASSERT_TRUE(report.ok());
  EXPECT_EQ(report.value().removed, 3U);
  EXPECT_EQ(report.value().failed, 0U);
  EXPECT_EQ(listing(d), (std::vector<std::string>{"MANIFEST", "index.000003.vfidx",
                                                  "index.abc.vfidx", "notes.txt"}));
}

TEST(Snapshot, DetectsMismatchedOrDamagedManifest) {
  const vf::test::ScopedTempDir dir("snapshot_bad");
  const auto a = make(40);
  const auto b = make(60);
  ASSERT_TRUE(vf::detail::save_snapshot(*a, dir.path()).ok());
  // Replace the current generation file by a different (valid) index: the manifest checksum no
  // longer matches.
  ASSERT_TRUE(b->save(dir.file("index.000001.vfidx")).ok());
  EXPECT_EQ(vf::detail::load_snapshot(dir.path()).status().code(), ErrorCode::CorruptData);
  std::ofstream(dir.file("MANIFEST"), std::ios::trunc) << "{\"format\": 1";
  EXPECT_EQ(vf::detail::load_snapshot(dir.path()).status().code(), ErrorCode::CorruptData);
  EXPECT_EQ(vf::detail::save_snapshot(*a, dir.path()).status().code(), ErrorCode::CorruptData)
      << "an unreadable MANIFEST is never overwritten blindly";
}

TEST(Snapshot, MappedGenerationSurvivesUntilReleased) {
  const vf::test::ScopedTempDir dir("snapshot_mapped");
  const auto state = make(40);
  ASSERT_TRUE(vf::detail::save_snapshot(*state, dir.path()).ok());
  auto mapped = vf::detail::load_snapshot(dir.path(), {.use_mmap = true}).value();
  ASSERT_TRUE(vf::detail::save_snapshot(*state, dir.path()).ok())
      << "a new generation can be published while the old one is mapped";
  EXPECT_EQ(mapped->search(std::vector<float>(kDim, 0.1F)).value().size(), 10U)
      << "the mapped generation stays readable";
  // POSIX and Windows 10+ (POSIX delete semantics; mappings open files with FILE_SHARE_DELETE)
  // unlink the mapped generation immediately. Where deletion fails, garbage collection reports
  // the file and retries on the next pass, which the release below exercises.
  mapped.reset();
  ASSERT_TRUE(vf::detail::collect_garbage(dir.path()).ok());
  EXPECT_EQ(listing(dir.path()), (std::vector<std::string>{"MANIFEST", "index.000002.vfidx"}));
}

}  // namespace
