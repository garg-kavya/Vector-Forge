// Catalog (docs/DESIGN.md §12.4): name validation, create/get/list/drop, snapshots, restart
// recovery (create, insert, snapshot, restart, search), durable drops, interrupted creations and
// damaged files.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <vectorforge/catalog.hpp>

#include "support/test_data.hpp"

namespace {

using vf::Catalog;
using vf::CollectionConfig;
using vf::ErrorCode;

CollectionConfig config(std::uint32_t dim, vf::IndexType index = vf::IndexType::Hnsw,
                        vf::Metric metric = vf::Metric::L2) {
  CollectionConfig c;
  c.dim = dim;
  c.index = index;
  c.metric = metric;
  c.hnsw.M = 6;
  c.hnsw.ef_construction = 40;
  c.hnsw.seed = 99;
  return c;
}

std::vector<vf::ExternalId> iota(std::size_t n) {
  std::vector<vf::ExternalId> ids(n);
  for (std::size_t i = 0; i < n; ++i) {
    ids[i] = i;
  }
  return ids;
}

TEST(Catalog, NameValidation) {
  EXPECT_TRUE(Catalog::is_valid_name("a"));
  EXPECT_TRUE(Catalog::is_valid_name("Docs_2024-v1"));
  EXPECT_TRUE(Catalog::is_valid_name(std::string(64, 'x')));
  const std::vector<std::string> bad_names{"",     std::string(65, 'x'), "a/b", "..", ".", "a b",
                                           "a\\b", "\xC3\xBC",           "a.b", "C:"};
  for (const std::string& bad : bad_names) {
    EXPECT_FALSE(Catalog::is_valid_name(bad)) << bad;
  }
}

TEST(Catalog, CreateGetListDrop) {
  const vf::test::ScopedTempDir dir("catalog");
  auto catalog = Catalog::open(dir.path()).value();
  EXPECT_EQ(catalog->size(), 0U);
  EXPECT_EQ(catalog->create("../evil", config(4)).status().code(), ErrorCode::InvalidArgument);
  CollectionConfig invalid = config(0);
  EXPECT_EQ(catalog->create("x", invalid).status().code(), ErrorCode::InvalidArgument);
  ASSERT_TRUE(catalog->create("b", config(4)).ok());
  ASSERT_TRUE(catalog->create("a", config(3, vf::IndexType::Flat)).ok());
  EXPECT_EQ(catalog->create("a", config(3)).status().code(), ErrorCode::AlreadyExists);
  EXPECT_EQ(catalog->list(), (std::vector<std::string>{"a", "b"}));
  EXPECT_EQ(catalog->get("a").value()->config().dim, 3U);
  EXPECT_EQ(catalog->get("zzz").status().code(), ErrorCode::NotFound);
  EXPECT_TRUE(std::filesystem::exists(dir.path() / "collections" / "a" / "config.json"));

  const auto entries = catalog->entries();
  ASSERT_EQ(entries.size(), 2U);
  EXPECT_EQ(entries[1].name, "b");
  EXPECT_EQ(entries[1].last_generation, 0U);

  ASSERT_TRUE(catalog->drop("a").ok());
  EXPECT_EQ(catalog->drop("a").code(), ErrorCode::NotFound);
  EXPECT_EQ(catalog->list(), (std::vector<std::string>{"b"}));
  EXPECT_FALSE(std::filesystem::exists(dir.path() / "collections" / "a"));
  ASSERT_TRUE(catalog->create("a", config(5)).ok());
  EXPECT_EQ(catalog->get("a").value()->config().dim, 5U);
}

TEST(Catalog, RestartRecovery) {
  const vf::test::ScopedTempDir dir("catalog_restart");
  const vf::test::ClusteredData data(7, 8, 5, 0.3F);
  const std::vector<float> rows = data.rows(1, 300);
  {
    auto catalog = Catalog::open(dir.path()).value();
    auto snap = catalog->create("snap", config(8)).value();
    ASSERT_TRUE(snap->add_batch(iota(300), rows).ok());
    const vf::Result<vf::SnapshotInfo> first = catalog->snapshot("snap");
    ASSERT_TRUE(first.ok()) << first.status().to_string();
    EXPECT_EQ(first.value().generation, 1U);
    EXPECT_GT(first.value().bytes, 0U);
    ASSERT_TRUE(snap->remove(3).ok());
    EXPECT_EQ(catalog->snapshot("snap").value().generation, 2U);
    EXPECT_EQ(catalog->entries()[0].last_generation, 2U);
    // Never snapshotted: comes back empty with its configuration.
    auto empty =
        catalog->create("empty", config(8, vf::IndexType::Flat, vf::Metric::InnerProduct)).value();
    ASSERT_TRUE(empty->add(1, std::vector<float>(8, 1.0F)).ok());
    EXPECT_EQ(catalog->snapshot("nope").status().code(), ErrorCode::NotFound);
  }
  for (const bool mmap : {true, false}) {
    auto catalog = Catalog::open(dir.path(), {.use_mmap = mmap}).value();
    ASSERT_EQ(catalog->list(), (std::vector<std::string>{"empty", "snap"}));
    auto snap = catalog->get("snap").value();
    EXPECT_EQ(snap->size(), 299U);
    EXPECT_FALSE(snap->contains(3));
    EXPECT_EQ(snap->config().hnsw.seed, 99U);
    vf::SearchParams p;
    p.k = 1;
    p.ef_search = 64;
    const auto hits = snap->search(std::span<const float>(rows).subspan(10 * 8, 8), p).value();
    ASSERT_EQ(hits.size(), 1U);
    EXPECT_EQ(hits[0].id, 10U);
    auto empty = catalog->get("empty").value();
    EXPECT_EQ(empty->size(), 0U);
    EXPECT_EQ(empty->config().index, vf::IndexType::Flat);
    EXPECT_EQ(empty->config().metric, vf::Metric::InnerProduct);
    // Snapshots continue from the stored generation.
    EXPECT_EQ(catalog->snapshot("snap").value().generation, mmap ? 3U : 4U);
  }
}

TEST(Catalog, DropIsDurableAndDeferredWhileInUse) {
  const vf::test::ScopedTempDir dir("catalog_drop");
  {
    auto catalog = Catalog::open(dir.path()).value();
    auto held = catalog->create("gone", config(2)).value();
    ASSERT_TRUE(catalog->snapshot("gone").ok());
    ASSERT_TRUE(catalog->drop("gone").ok());
    EXPECT_TRUE(std::filesystem::exists(dir.path() / "collections" / "gone" / "DROPPED"));
    EXPECT_EQ(catalog->create("gone", config(2)).status().code(), ErrorCode::Unavailable);
    EXPECT_TRUE(held->add(1, std::vector<float>{1, 2}).ok());
    // `held` outlives the catalog; simulate a crash by leaving the directory in place.
    catalog.reset();
    EXPECT_TRUE(std::filesystem::exists(dir.path() / "collections" / "gone"));
    // Releasing the last reference deletes the files even though the catalog is gone.
    held.reset();
    EXPECT_FALSE(std::filesystem::exists(dir.path() / "collections" / "gone"));
  }
  // A marked directory left by a crash is removed on open.
  std::filesystem::create_directories(dir.path() / "collections" / "zombie");
  std::ofstream(dir.path() / "collections" / "zombie" / "DROPPED") << "dropped\n";
  // So is an interrupted creation.
  std::filesystem::create_directories(dir.path() / "collections" / ".half.creating");
  auto catalog = Catalog::open(dir.path()).value();
  EXPECT_TRUE(catalog->list().empty());
  EXPECT_FALSE(std::filesystem::exists(dir.path() / "collections" / "zombie"));
  EXPECT_FALSE(std::filesystem::exists(dir.path() / "collections" / ".half.creating"));
}

TEST(Catalog, DamagedFilesAreReported) {
  const vf::test::ScopedTempDir dir("catalog_bad");
  {
    auto catalog = Catalog::open(dir.path()).value();
    ASSERT_TRUE(catalog->create("broken", config(2)).ok());
  }
  std::ofstream(dir.path() / "collections" / "broken" / "config.json") << R"({"format": 1})";
  const auto opened = Catalog::open(dir.path());
  ASSERT_FALSE(opened.ok());
  EXPECT_EQ(opened.status().code(), ErrorCode::CorruptData);
  EXPECT_NE(opened.status().message().find("broken"), std::string::npos);

  std::ofstream(dir.path() / "collections" / "broken" / "MANIFEST") << "garbage";
  EXPECT_EQ(Catalog::open(dir.path()).status().code(), ErrorCode::CorruptData);
}

TEST(Catalog, ConfigRoundTripsEveryField) {
  const vf::test::ScopedTempDir dir("catalog_cfg");
  CollectionConfig c;
  c.dim = 17;
  c.metric = vf::Metric::Cosine;
  c.normalize = true;
  c.index = vf::IndexType::Hnsw;
  c.hnsw.M = 31;
  c.hnsw.ef_construction = 77;
  c.hnsw.ef_search = 13;
  c.hnsw.max_level = 9;
  c.hnsw.seed = 18446744073709551615ULL;
  {
    auto catalog = Catalog::open(dir.path()).value();
    ASSERT_TRUE(catalog->create("cfg", c).ok());
  }
  auto catalog = Catalog::open(dir.path(), {.concurrency = vf::Concurrency::Coarse}).value();
  const CollectionConfig r = catalog->get("cfg").value()->config();
  EXPECT_EQ(r.dim, c.dim);
  EXPECT_EQ(r.metric, c.metric);
  EXPECT_EQ(r.normalize, c.normalize);
  EXPECT_EQ(r.index, c.index);
  EXPECT_EQ(r.hnsw.M, c.hnsw.M);
  EXPECT_EQ(r.hnsw.ef_construction, c.hnsw.ef_construction);
  EXPECT_EQ(r.hnsw.ef_search, c.hnsw.ef_search);
  EXPECT_EQ(r.hnsw.max_level, c.hnsw.max_level);
  EXPECT_EQ(r.hnsw.seed, c.hnsw.seed);
  EXPECT_EQ(r.concurrency, vf::Concurrency::Coarse);
}

}  // namespace
