// Damaged and hostile index files (docs/DESIGN.md §12.3): truncation at every section boundary and
// at random offsets, single-bit flips of every byte, and semantically hostile values with valid
// checksums. The reader must return an error (never crash, assert or read out of bounds; ASan and
// UBSan builds check the latter). With full verification every flip must be detected.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <vectorforge/collection.hpp>

#include "collection/collection_factory.hpp"
#include "core/rng.hpp"
#include "storage/format.hpp"
#include "support/persistence.hpp"
#include "support/test_data.hpp"
#include "support/vfidx_editor.hpp"

namespace {

using vf::Collection;
using vf::ErrorCode;
using vf::Verify;
using vf::detail::CollectionFactory;
using ST = vf::detail::format::SectionType;
namespace fmt = vf::detail::format;

constexpr std::uint32_t kDim = 4;
constexpr std::uint32_t kM = 4;

// HNSW cosine collection with tombstones and upserts: every section type is present.
std::vector<std::byte> make_hnsw_file() {
  vf::CollectionConfig cfg =
      vf::test::small_config(vf::IndexType::Hnsw, vf::Metric::Cosine, false, kDim);
  cfg.hnsw.M = kM;
  cfg.hnsw.ef_construction = 16;
  const auto c = Collection::create(cfg).value();
  vf::test::apply_operations(*c, 7, 200, 0);
  return vf::test::save_to_memory(*c);
}

std::vector<std::byte> make_flat_file() {
  const auto c =
      Collection::create(vf::test::small_config(vf::IndexType::Flat, vf::Metric::L2, false, kDim))
          .value();
  vf::test::apply_operations(*c, 8, 100, 0);
  return vf::test::save_to_memory(*c);
}

vf::Result<std::unique_ptr<Collection>> load(std::span<const std::byte> bytes, Verify verify) {
  return CollectionFactory::load_from_memory(bytes, verify);
}

// A file that loads must be fully usable: searches run and its re-encoding verifies.
void exercise(const Collection& c) {
  const std::vector<float> q(kDim, 0.5F);
  vf::SearchParams p;
  p.k = 5;
  static_cast<void>(c.search(q, p));
  const std::vector<std::byte> resaved = vf::test::save_to_memory(c);
  const auto again = load(resaved, Verify::Full);
  ASSERT_TRUE(again.ok()) << "re-encoding of a loaded file must verify: "
                          << again.status().to_string();
}

TEST(Corruption, IntactFilesLoad) {
  for (const auto& bytes : {make_hnsw_file(), make_flat_file()}) {
    for (const Verify v : {Verify::None, Verify::Metadata, Verify::Full}) {
      const auto c = load(bytes, v);
      ASSERT_TRUE(c.ok()) << c.status().to_string();
      exercise(*c.value());
    }
  }
}

TEST(Corruption, TruncationAtBoundariesAndRandomOffsets) {
  const std::vector<std::byte> bytes = make_hnsw_file();
  const vf::test::VfidxEditor editor(bytes);
  std::vector<std::size_t> cuts = {0, 1, 8, 12, 55, 63, 64, 65};
  for (const fmt::SectionEntry& e : editor.entries()) {
    for (const std::uint64_t at : {e.offset, e.offset + e.size}) {
      for (const std::int64_t delta : {-1, 0, 1}) {
        cuts.push_back(static_cast<std::size_t>(static_cast<std::int64_t>(at) + delta));
      }
    }
  }
  cuts.push_back(static_cast<std::size_t>(editor.table_offset()));
  cuts.push_back(bytes.size() - 1);
  vf::detail::Xoshiro256ss rng(3);
  for (int i = 0; i < 200; ++i) {
    cuts.push_back(static_cast<std::size_t>(vf::detail::uniform_below(rng, bytes.size())));
  }
  const vf::test::ScopedTempDir dir("truncate");
  for (const std::size_t cut : cuts) {
    if (cut >= bytes.size()) {
      continue;
    }
    SCOPED_TRACE(testing::Message() << "truncated to " << cut << " of " << bytes.size());
    const auto prefix = std::span<const std::byte>(bytes).first(cut);
    for (const Verify v : {Verify::None, Verify::Metadata, Verify::Full}) {
      EXPECT_EQ(load(prefix, v).status().code(), ErrorCode::CorruptData);
    }
    vf::test::write_bytes(dir.file("t.vfidx"), prefix);
    EXPECT_EQ(Collection::load(dir.file("t.vfidx"), {.use_mmap = true, .verify = Verify::None})
                  .status()
                  .code(),
              ErrorCode::CorruptData);
  }
  // Appending bytes is detected as well.
  std::vector<std::byte> longer = bytes;
  longer.push_back(std::byte{0});
  EXPECT_EQ(load(longer, Verify::None).status().code(), ErrorCode::CorruptData);
}

TEST(Corruption, EverySingleBitFlipIsDetectedWithFullVerification) {
  for (const auto& original : {make_hnsw_file(), make_flat_file()}) {
    std::vector<std::byte> bytes = original;
    std::size_t loaded_without_crc = 0;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
      const auto bit = static_cast<unsigned>(i % 8);
      bytes[i] ^= static_cast<std::byte>(1U << bit);
      const auto full = load(bytes, Verify::Full);
      EXPECT_FALSE(full.ok()) << "flip of bit " << bit << " at byte " << i << " went undetected";
      if (i % 3 == 0) {  // weaker verification: must not crash; whatever loads must be usable
        for (const Verify v : {Verify::None, Verify::Metadata}) {
          const auto weak = load(bytes, v);
          if (weak.ok()) {
            ++loaded_without_crc;
            exercise(*weak.value());
          }
        }
      }
      bytes[i] = original[i];
    }
    EXPECT_GT(loaded_without_crc, 0U) << "some vector-byte flips load without checksums";
  }
}

class HostileFile : public testing::Test {
 protected:
  void SetUp() override { bytes_ = make_hnsw_file(); }

  // Applies `edit`, recomputes checksums, and expects the load to fail with `code`.
  void expect_rejected(const std::function<void(vf::test::VfidxEditor&)>& edit, ErrorCode code,
                       const std::string& what, bool fix = true) const {
    vf::test::VfidxEditor editor(bytes_);
    edit(editor);
    if (fix) {
      editor.fix_checksums();
    }
    for (const Verify v : {Verify::None, Verify::Full}) {
      const auto result = load(editor.bytes(), v);
      EXPECT_EQ(result.status().code(), code)
          << what << " (verify " << static_cast<int>(v) << "): " << result.status().to_string();
    }
  }

  [[nodiscard]] std::uint64_t node_count() const {
    return vf::test::VfidxEditor(bytes_).get(
        vf::test::VfidxEditor(bytes_).offset_of(ST::Metadata) + 8, 8);
  }

  std::vector<std::byte> bytes_;
};

TEST_F(HostileFile, HeaderAndTable) {
  const auto corrupt = ErrorCode::CorruptData;
  expect_rejected([](auto& e) { e.put(0, 0x58444946, 4); }, corrupt, "bad magic");
  expect_rejected([](auto& e) { e.put(12, fmt::kEndianTagSwapped, 4); },
                  ErrorCode::UnsupportedVersion, "big-endian");
  expect_rejected([](auto& e) { e.put(12, 7, 4); }, corrupt, "bad endian tag");
  expect_rejected([](auto& e) { e.put(8, 2, 2); }, ErrorCode::UnsupportedVersion, "major 2");
  expect_rejected([](auto& e) { e.put(16, 1U << 20U, 8); }, corrupt, "unknown flag");
  expect_rejected([](auto& e) { e.put(16, e.get(16, 8) & ~fmt::kFlagHasGraph, 8); }, corrupt,
                  "HAS_GRAPH cleared");
  expect_rejected([](auto& e) { e.put(16, e.get(16, 8) & ~fmt::kFlagHasTombstones, 8); }, corrupt,
                  "HAS_TOMBSTONES cleared");
  expect_rejected([](auto& e) { e.put(16, e.get(16, 8) & ~fmt::kFlagVectorsNormalized, 8); },
                  corrupt, "normalized flag cleared");
  expect_rejected([](auto& e) { e.put(24, e.get(24, 8) + 8, 8); }, corrupt, "file_size too large");
  expect_rejected([](auto& e) { e.put(40, 0, 4); }, corrupt, "no sections");
  expect_rejected([](auto& e) { e.put(40, 1000, 4); }, corrupt, "too many sections");
  expect_rejected([](auto& e) { e.put(32, std::numeric_limits<std::uint64_t>::max() - 4, 8); },
                  corrupt, "table offset overflow");
  expect_rejected([](auto& e) { e.put(48, 1, 8); }, corrupt, "reserved header field");
  expect_rejected([](auto& e) { e.put(56, e.get(56, 4) ^ 1U, 4); }, corrupt, "header checksum",
                  /*fix=*/false);
  expect_rejected([](auto& e) { e.put(44, e.get(44, 4) ^ 1U, 4); }, corrupt, "table checksum",
                  /*fix=*/false);
}

TEST_F(HostileFile, SectionBoundsOverlapAndTypes) {
  const auto corrupt = ErrorCode::CorruptData;
  auto edit_entry = [](ST type, const std::function<void(fmt::SectionEntry&)>& change) {
    return [type, change](vf::test::VfidxEditor& e) {
      const auto i = static_cast<std::size_t>(e.entry_index(type));
      fmt::SectionEntry entry = e.entries()[i];
      change(entry);
      e.put_entry(i, entry);
    };
  };
  expect_rejected(edit_entry(ST::Labels, [](auto& s) { s.offset = std::uint64_t{1} << 40U; }),
                  corrupt, "section beyond EOF");
  expect_rejected(
      edit_entry(ST::Labels, [](auto& s) { s.size = std::numeric_limits<std::uint64_t>::max(); }),
      corrupt, "size overflow");
  expect_rejected(edit_entry(ST::Labels, [](auto& s) { s.offset = 64; }), corrupt,
                  "overlapping sections");
  expect_rejected(edit_entry(ST::Labels, [](auto& s) { s.offset += 3; }), corrupt,
                  "misaligned offset");
  expect_rejected(
      edit_entry(ST::Labels, [](auto& s) { s.type = static_cast<std::uint32_t>(ST::Levels); }),
      corrupt, "duplicate type");
  expect_rejected(edit_entry(ST::Labels, [](auto& s) { s.type = 99; }),
                  ErrorCode::UnsupportedVersion, "unknown required section");
  expect_rejected(edit_entry(ST::Labels, [](auto& s) { s.flags = 2; }), corrupt,
                  "unknown section flag");
  expect_rejected(edit_entry(ST::Tombstones,
                             [](auto& s) {
                               s.type = 99;
                               s.flags = fmt::kSectionOptional;
                             }),
                  corrupt, "tombstones missing while flagged (unknown optional section skipped)");
  expect_rejected(edit_entry(ST::Labels, [](auto& s) { s.size -= 8; }), corrupt, "labels size");
  expect_rejected([](auto& e) { e.put(e.offset_of(ST::Vectors) - 1, 1, 1); }, corrupt,
                  "non-zero padding before VECTORS");
  expect_rejected([](auto& e) { e.put(e.table_offset() + 28, 1, 1); }, corrupt,
                  "table entry reserved field");
}

TEST_F(HostileFile, MetadataValues) {
  const auto corrupt = ErrorCode::CorruptData;
  const std::uint64_t n = node_count();
  expect_rejected([](auto& e) { e.put_metadata(0, 0, 4); }, corrupt, "dim 0");
  expect_rejected([](auto& e) { e.put_metadata(0, 70000, 4); }, corrupt, "dim too large");
  expect_rejected([](auto& e) { e.put_metadata(0, kDim + 1, 4); }, corrupt, "dim/size mismatch");
  expect_rejected([](auto& e) { e.put_metadata(4, 9, 1); }, corrupt, "metric");
  expect_rejected([](auto& e) { e.put_metadata(5, 7, 1); }, corrupt, "index type");
  expect_rejected([](auto& e) { e.put_metadata(6, 2, 1); }, corrupt, "normalize");
  expect_rejected([](auto& e) { e.put_metadata(5, 0, 1); }, corrupt, "flat index with graph");
  expect_rejected([](auto& e) { e.put_metadata(8, std::uint64_t{1} << 40U, 8); }, corrupt,
                  "node_count huge");
  expect_rejected([](auto& e) { e.put_metadata(8, 0xFFFFFFFEULL, 8); }, corrupt,
                  "node_count * dim");
  expect_rejected([n](auto& e) { e.put_metadata(16, n + 1, 8); }, corrupt,
                  "live_count > node_count");
  expect_rejected([n](auto& e) { e.put_metadata(16, n, 8); }, corrupt, "live_count vs tombstones");
  expect_rejected([](auto& e) { e.put_metadata(24, 1, 4); }, corrupt, "M = 1");
  expect_rejected([](auto& e) { e.put_metadata(28, 2 * kM + 1, 4); }, corrupt, "M0 != 2M");
  expect_rejected([](auto& e) { e.put_metadata(32, 1, 4); }, corrupt, "ef_construction < M");
  expect_rejected([](auto& e) { e.put_metadata(40, 40, 1); }, corrupt, "max_level_cap > 32");
  expect_rejected(
      [](auto& e) { e.put_metadata(41, e.get(e.offset_of(ST::Metadata) + 40, 1) + 1, 1); }, corrupt,
      "max_level > cap");
  expect_rejected([n](auto& e) { e.put_metadata(44, n, 4); }, corrupt, "entry point out of range");
  const vf::test::VfidxEditor probe(bytes_);
  std::uint64_t level0_node = 0;
  while (probe.get(probe.offset_of(ST::Levels) + level0_node, 1) != 0) {
    ++level0_node;
  }
  expect_rejected([level0_node](auto& e) { e.put_metadata(44, level0_node, 4); }, corrupt,
                  "entry point not on the top level");
  expect_rejected([](auto& e) { e.put_metadata(56, 1, 8); }, corrupt, "reserved field");
  expect_rejected([](auto& e) { e.put_metadata(64, 1000, 2); }, corrupt, "creator length");
}

TEST_F(HostileFile, GraphLabelsTombstonesVectors) {
  const auto corrupt = ErrorCode::CorruptData;
  const std::uint64_t n = node_count();
  const std::uint64_t stride0 = (2 * kM + 1) * 4;
  // Find a level-0 list with at least two links.
  const vf::test::VfidxEditor probe(bytes_);
  std::uint64_t node = 0;
  auto list_count = [&](std::uint64_t i) {
    return probe.get(probe.offset_of(ST::L0Links) + (i * stride0), 4);
  };
  while (list_count(node) < 2 || list_count(node) >= 2 * kM) {
    ++node;
  }
  const std::uint64_t list = probe.offset_of(ST::L0Links) + (node * stride0);
  const std::uint64_t count = probe.get(list, 4);
  expect_rejected([list](auto& e) { e.put(list, 2 * kM + 1, 4); }, corrupt, "count > capacity");
  expect_rejected([list, n](auto& e) { e.put(list + 4, n, 4); }, corrupt,
                  "link to nonexistent node");
  expect_rejected([list, node](auto& e) { e.put(list + 4, node, 4); }, corrupt, "self link");
  expect_rejected([list](auto& e) { e.put(list + 8, e.get(list + 4, 4), 4); }, corrupt,
                  "duplicate link");
  expect_rejected([list, count](auto& e) { e.put(list + 4 + (4 * count), 0, 4); }, corrupt,
                  "non-empty unused slot");
  // Upper levels: the first node with a block links to something; point it at a level-0 node.
  std::uint64_t level0_node = 0;
  while (probe.get(probe.offset_of(ST::Levels) + level0_node, 1) != 0) {
    ++level0_node;
  }
  std::uint64_t upper_node = 0;
  while (probe.get(probe.offset_of(ST::UpperIndex) + (upper_node * 8), 8) == fmt::kNoUpperBlock) {
    ++upper_node;
  }
  const std::uint64_t block = probe.offset_of(ST::UpperLinks);
  expect_rejected(
      [block, level0_node](auto& e) {
        e.put(block, 1, 4);
        e.put(block + 4, level0_node, 4);
      },
      corrupt, "upper-level link to a level-0 node");
  expect_rejected(
      [upper_node](auto& e) { e.put(e.offset_of(ST::UpperIndex) + (upper_node * 8), 4, 8); },
      corrupt, "non-canonical upper offset");
  expect_rejected([level0_node](auto& e) { e.put(e.offset_of(ST::Levels) + level0_node, 33, 1); },
                  corrupt, "level above cap");
  expect_rejected([level0_node](auto& e) { e.put(e.offset_of(ST::Levels) + level0_node, 1, 1); },
                  corrupt, "levels inconsistent with UPPER_LINKS size");

  // Labels: find two live rows (tombstone bit clear) and give them the same label.
  const std::uint64_t tomb = probe.offset_of(ST::Tombstones);
  std::vector<std::uint64_t> live;
  for (std::uint64_t r = 0; r < n && live.size() < 2; ++r) {
    if (((probe.get(tomb + ((r / 64) * 8), 8) >> (r % 64)) & 1U) == 0U) {
      live.push_back(r);
    }
  }
  const std::uint64_t labels = probe.offset_of(ST::Labels);
  expect_rejected(
      [&](auto& e) { e.put(labels + (live[1] * 8), e.get(labels + (live[0] * 8), 8), 8); }, corrupt,
      "duplicate live label");
  expect_rejected([&](auto& e) { e.put(labels + (live[0] * 8), vf::kInvalidExternalId, 8); },
                  corrupt, "reserved label");
  expect_rejected(
      [tomb, n](auto& e) {
        const std::uint64_t last_word = tomb + (((n - 1) / 64) * 8);
        e.put(last_word, e.get(last_word, 8) | (std::uint64_t{1} << 63U), 8);
      },
      corrupt, "tombstone bit beyond the last row");

  // Vectors: a NaN component is rejected when vectors are copied or fully verified.
  const std::uint64_t vec = probe.offset_of(ST::Vectors);
  expect_rejected([vec](auto& e) { e.put(vec, 0x7FC00000U, 4); }, corrupt, "NaN component");
  expect_rejected([vec](auto& e) { e.put(vec + 4, 0x7F800000U, 4); }, corrupt,
                  "infinite component");
}

}  // namespace
