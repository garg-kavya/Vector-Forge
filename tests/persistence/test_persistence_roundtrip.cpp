// Save/load round trips (docs/DESIGN.md §15.2 "Persistence"): identical configuration, contents
// and bit-identical search results in every load mode; save -> load -> save is byte-identical; a
// loaded collection keeps evolving exactly like the original.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <vectorforge/collection.hpp>

#include "support/persistence.hpp"
#include "support/test_data.hpp"

namespace {

using vf::Collection;
using vf::ErrorCode;
using vf::IndexType;
using vf::LoadOptions;
using vf::Metric;
using vf::Verify;

struct RoundTripCase {
  IndexType index;
  Metric metric;
  bool normalize;
};

std::string case_name(const testing::TestParamInfo<RoundTripCase>& info) {
  return std::string(vf::to_string(info.param.index)) + "_" +
         std::string(vf::to_string(info.param.metric)) +
         (info.param.normalize ? "_normalized" : "");
}

class PersistenceRoundTrip : public testing::TestWithParam<RoundTripCase> {};

constexpr std::uint32_t kDim = 12;

const LoadOptions kModes[] = {
    {.use_mmap = false, .verify = Verify::Auto},
    {.use_mmap = false, .verify = Verify::None},
    {.use_mmap = true, .verify = Verify::Auto},
    {.use_mmap = true, .verify = Verify::Full},
    {.use_mmap = true, .verify = Verify::None, .prefault = true},
};

TEST_P(PersistenceRoundTrip, LoadIsIdenticalInEveryMode) {
  const RoundTripCase rc = GetParam();
  const vf::test::ScopedTempDir dir("roundtrip");
  const std::unique_ptr<Collection> original =
      Collection::create(vf::test::small_config(rc.index, rc.metric, rc.normalize, kDim)).value();
  vf::test::apply_operations(*original, 1, 1500, 0);
  const vf::CollectionStats before = original->stats();
  ASSERT_GT(before.deleted_count, 0U);

  const std::filesystem::path file = dir.file("a.vfidx");
  ASSERT_TRUE(original->save(file).ok());
  const std::vector<std::byte> bytes = vf::test::read_bytes(file);
  EXPECT_EQ(bytes, vf::test::save_to_memory(*original)) << "file and in-memory encodings agree";
  ASSERT_TRUE(original->save(dir.file("again.vfidx")).ok());
  EXPECT_EQ(vf::test::read_bytes(dir.file("again.vfidx")), bytes) << "saving is deterministic";

  const vf::test::ClusteredData data(5, kDim, 12, 0.3F);
  const std::vector<float> queries = data.rows(99, 40);
  for (std::size_t mode = 0; mode < std::size(kModes); ++mode) {
    SCOPED_TRACE(testing::Message() << "mode " << mode);
    const vf::Result<std::unique_ptr<Collection>> loaded = Collection::load(file, kModes[mode]);
    ASSERT_TRUE(loaded.ok()) << loaded.status().to_string();
    const Collection& c = *loaded.value();
    vf::test::expect_same_config(c.config(), original->config());
    const vf::CollectionStats after = c.stats();
    EXPECT_EQ(after.live_count, before.live_count);
    EXPECT_EQ(after.deleted_count, before.deleted_count);
    EXPECT_EQ(after.row_count, before.row_count);
    EXPECT_EQ(after.normalized, before.normalized);
    // These collections are smaller than one vector chunk, so even mmap loads copy the (partial)
    // chunk; MmapServesFullChunksInPlace covers mapped chunks.
    EXPECT_EQ(after.memory.mapped_vectors_bytes, 0U);
    vf::test::expect_same_results(*original, c, queries, 10);
    for (vf::ExternalId id = 1000; id < 1000 + (3 * 1500); id += 3) {
      ASSERT_EQ(c.contains(id), original->contains(id)) << "id " << id;
      if (original->contains(id)) {
        ASSERT_EQ(c.get(id).value(), original->get(id).value()) << "id " << id;
      }
    }
    const std::filesystem::path resaved = dir.file("resaved_" + std::to_string(mode) + ".vfidx");
    ASSERT_TRUE(c.save(resaved).ok());
    EXPECT_EQ(vf::test::read_bytes(resaved), bytes) << "save -> load -> save is byte-identical";
  }
}

TEST_P(PersistenceRoundTrip, LoadedCollectionEvolvesLikeTheOriginal) {
  const RoundTripCase rc = GetParam();
  const vf::test::ScopedTempDir dir("evolve");
  const std::unique_ptr<Collection> original =
      Collection::create(vf::test::small_config(rc.index, rc.metric, rc.normalize, kDim)).value();
  vf::test::apply_operations(*original, 2, 700, 0);
  const std::filesystem::path file = dir.file("base.vfidx");
  ASSERT_TRUE(original->save(file).ok());

  for (const bool mmap : {false, true}) {
    SCOPED_TRACE(mmap ? "mmap" : "heap");
    std::unique_ptr<Collection> loaded = Collection::load(file, {.use_mmap = mmap}).value();
    const std::unique_ptr<Collection> twin = Collection::load(file, {.use_mmap = false}).value();
    // Continue both with identical operations: appends after the mapped base, upserts that
    // tombstone mapped rows, removals.
    vf::test::apply_operations(*loaded, 3, 400, 700);
    vf::test::apply_operations(*twin, 3, 400, 700);
    EXPECT_EQ(vf::test::save_to_memory(*loaded), vf::test::save_to_memory(*twin));
    const vf::test::ClusteredData data(5, kDim, 12, 0.3F);
    vf::test::expect_same_results(*loaded, *twin, data.rows(98, 30), 10);
  }
  // The original, evolved the same way, is indistinguishable from a loaded copy.
  std::unique_ptr<Collection> loaded = Collection::load(file).value();
  vf::test::apply_operations(*loaded, 3, 400, 700);
  vf::test::apply_operations(*original, 3, 400, 700);
  EXPECT_EQ(vf::test::save_to_memory(*loaded), vf::test::save_to_memory(*original));
}

INSTANTIATE_TEST_SUITE_P(
    Configs, PersistenceRoundTrip,
    testing::Values(RoundTripCase{IndexType::Flat, Metric::L2, false},
                    RoundTripCase{IndexType::Flat, Metric::InnerProduct, false},
                    RoundTripCase{IndexType::Flat, Metric::Cosine, false},
                    RoundTripCase{IndexType::Hnsw, Metric::L2, false},
                    RoundTripCase{IndexType::Hnsw, Metric::InnerProduct, false},
                    RoundTripCase{IndexType::Hnsw, Metric::Cosine, false},
                    RoundTripCase{IndexType::Hnsw, Metric::L2, true}),
    case_name);

TEST(Persistence, MmapServesFullChunksInPlace) {
  // dim 8192 -> 512 rows per 16 MiB chunk: 1100 rows = two mapped chunks plus a copied tail.
  constexpr std::uint32_t kWide = 8192;
  const vf::test::ScopedTempDir dir("mmap_chunks");
  const std::unique_ptr<Collection> original =
      Collection::create(vf::test::small_config(IndexType::Flat, Metric::L2, false, kWide)).value();
  vf::detail::Xoshiro256ss rng(21);
  for (vf::ExternalId id = 0; id < 1100; ++id) {
    ASSERT_TRUE(original->add(id, vf::test::random_vector(rng, kWide)).ok());
  }
  ASSERT_TRUE(original->remove(5).ok());
  const std::filesystem::path file = dir.file("wide.vfidx");
  ASSERT_TRUE(original->save(file).ok());

  const auto mapped = Collection::load(file, {.use_mmap = true, .verify = Verify::Full}).value();
  const vf::detail::CollectionState& s = vf::detail::CollectionFactory::state(*mapped);
  EXPECT_EQ(s.vectors.mapped_rows(), 1024U);
  EXPECT_EQ(mapped->stats().memory.mapped_vectors_bytes, std::size_t{1024} * kWide * 4);
  EXPECT_EQ(mapped->stats().memory.vectors_bytes, s.vectors.chunk_bytes())
      << "one heap chunk for the tail";
  const std::vector<float> queries = vf::test::random_matrix(rng, 4, kWide);
  vf::test::expect_same_results(*original, *mapped, queries, 5);
  for (vf::ExternalId id = 1100; id < 1200; ++id) {  // appends after the mapped base
    const std::vector<float> v = vf::test::random_vector(rng, kWide);
    ASSERT_TRUE(mapped->add(id, v).ok());
    ASSERT_TRUE(original->add(id, v).ok());
  }
  ASSERT_TRUE(mapped->add(7, vf::test::random_vector(rng, kWide), {.upsert = true}).ok())
      << "upserting a mapped row tombstones it";
  EXPECT_EQ(mapped->get(1000).value(), original->get(1000).value());
  EXPECT_FALSE(mapped->contains(5));
  const auto heap = Collection::load(file, {.use_mmap = false}).value();
  EXPECT_EQ(vf::detail::CollectionFactory::state(*heap).vectors.mapped_rows(), 0U);
  EXPECT_EQ(heap->stats().memory.mapped_vectors_bytes, 0U);
}

TEST(Persistence, EmptyCollections) {
  const vf::test::ScopedTempDir dir("empty");
  for (const IndexType index : {IndexType::Flat, IndexType::Hnsw}) {
    SCOPED_TRACE(vf::to_string(index));
    const std::unique_ptr<Collection> c =
        Collection::create(vf::test::small_config(index, Metric::L2, false, 5)).value();
    const std::filesystem::path file = dir.file(std::string(vf::to_string(index)) + ".vfidx");
    ASSERT_TRUE(c->save(file).ok());
    for (const bool mmap : {false, true}) {
      auto loaded = Collection::load(file, {.use_mmap = mmap});
      ASSERT_TRUE(loaded.ok()) << loaded.status().to_string();
      EXPECT_EQ(loaded.value()->size(), 0U);
      EXPECT_TRUE(loaded.value()->search(std::vector<float>(5, 1.0F)).value().empty());
      ASSERT_TRUE(loaded.value()->add(1, std::vector<float>(5, 1.0F)).ok());
      EXPECT_EQ(loaded.value()->search(std::vector<float>(5, 1.0F)).value().size(), 1U);
    }
    EXPECT_EQ(vf::test::read_bytes(file), vf::test::save_to_memory(*c));
  }
}

TEST(Persistence, ProvenanceIsPreserved) {
  const vf::test::ScopedTempDir dir("provenance");
  const std::unique_ptr<Collection> c =
      Collection::create(vf::test::small_config(IndexType::Flat, Metric::L2, false, 3)).value();
  const std::filesystem::path file = dir.file("p.vfidx");
  ASSERT_TRUE(c->save(file).ok());
  const auto loaded = Collection::load(file).value();
  const vf::detail::CollectionState& a = vf::detail::CollectionFactory::state(*c);
  const vf::detail::CollectionState& b = vf::detail::CollectionFactory::state(*loaded);
  EXPECT_EQ(a.creator, b.creator);
  EXPECT_EQ(a.created_unix_ms, b.created_unix_ms);
  EXPECT_NE(a.created_unix_ms, 0U);
  EXPECT_EQ(a.creator.rfind("vectorforge ", 0), 0U);
}

TEST(Persistence, LoadErrors) {
  const vf::test::ScopedTempDir dir("load_errors");
  EXPECT_EQ(Collection::load(dir.file("missing.vfidx")).status().code(), ErrorCode::IoError);
  vf::test::write_bytes(dir.file("empty.vfidx"), {});
  EXPECT_EQ(Collection::load(dir.file("empty.vfidx")).status().code(), ErrorCode::CorruptData);
  EXPECT_EQ(Collection::load(dir.file("empty.vfidx"), {.use_mmap = false}).status().code(),
            ErrorCode::CorruptData);
}

TEST(Persistence, SavingOverTheMappedSourceFile) {
  const vf::test::ScopedTempDir dir("overwrite");
  const std::unique_ptr<Collection> c =
      Collection::create(vf::test::small_config(IndexType::Flat, Metric::L2, false, 4)).value();
  ASSERT_TRUE(c->add(1, std::vector<float>{1, 2, 3, 4}).ok());
  const std::filesystem::path file = dir.file("self.vfidx");
  ASSERT_TRUE(c->save(file).ok());
  const auto mapped = Collection::load(file, {.use_mmap = true}).value();
  ASSERT_TRUE(mapped->add(2, std::vector<float>{5, 6, 7, 8}).ok());
  const vf::Status st = mapped->save(file);
#if defined(_WIN32)
  // Windows cannot replace a file with an active mapping; the old file stays intact.
  EXPECT_EQ(st.code(), ErrorCode::IoError);
  EXPECT_EQ(Collection::load(file).value()->size(), 1U);
#else
  // POSIX renames over the mapped inode; the mapping keeps reading the old data.
  ASSERT_TRUE(st.ok()) << st.to_string();
  EXPECT_EQ(mapped->get(1).value(), (std::vector<float>{1, 2, 3, 4}));
  EXPECT_EQ(Collection::load(file).value()->size(), 2U);
#endif
}

}  // namespace
