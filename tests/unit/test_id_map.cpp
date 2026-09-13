#include <gtest/gtest.h>

#include <cstdint>

#include <vectorforge/types.hpp>

#include "storage/id_map.hpp"
#include "storage/tombstones.hpp"

namespace {

using vf::ErrorCode;
using vf::kInvalidInternalId;
using vf::detail::IdMap;
using vf::detail::TombstoneSet;

TEST(IdMap, ReserveCommitFind) {
  IdMap map;
  map.reserve_capacity(4);
  auto r = map.reserve(42, /*upsert=*/false);
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.value().previous, kInvalidInternalId);
  EXPECT_FALSE(map.contains(42)) << "pending new keys are invisible";
  EXPECT_EQ(map.size(), 0U);

  EXPECT_EQ(map.commit(r.value(), 0), kInvalidInternalId);
  EXPECT_EQ(map.find(42), 0U);
  EXPECT_TRUE(map.contains(42));
  EXPECT_EQ(map.size(), 1U);
  EXPECT_EQ(map.label_count(), 1U);
  EXPECT_EQ(map.label(0), 42U);
}

TEST(IdMap, DuplicateAndReservedKeys) {
  IdMap map;
  map.reserve_capacity(2);
  auto r = map.reserve(7, false);
  ASSERT_TRUE(r.ok());
  // Second reservation while pending, even with upsert, is rejected.
  EXPECT_EQ(map.reserve(7, true).status().code(), ErrorCode::AlreadyExists);
  static_cast<void>(map.commit(r.value(), 0));
  EXPECT_EQ(map.reserve(7, false).status().code(), ErrorCode::AlreadyExists);
  EXPECT_EQ(map.reserve(vf::kInvalidExternalId, true).status().code(), ErrorCode::InvalidArgument);
}

TEST(IdMap, UpsertReplacesAndKeepsOldVisibleUntilCommit) {
  IdMap map;
  map.reserve_capacity(3);
  static_cast<void>(map.commit(map.reserve(5, false).value(), 0));
  auto up = map.reserve(5, /*upsert=*/true);
  ASSERT_TRUE(up.ok());
  EXPECT_EQ(up.value().previous, 0U);
  EXPECT_EQ(map.find(5), 0U) << "old mapping stays visible during an upsert";
  EXPECT_EQ(map.commit(up.value(), 1), 0U) << "commit returns the replaced internal id";
  EXPECT_EQ(map.find(5), 1U);
  EXPECT_EQ(map.size(), 1U);
  EXPECT_EQ(map.label_count(), 2U);
  EXPECT_EQ(map.label(0), 5U) << "labels of replaced rows are kept";
  EXPECT_EQ(map.label(1), 5U);
}

TEST(IdMap, RollbackRestoresPreviousState) {
  IdMap map;
  map.reserve_capacity(3);
  auto fresh = map.reserve(1, false);
  map.rollback(fresh.value());
  EXPECT_FALSE(map.contains(1));
  EXPECT_TRUE(map.reserve(1, false).ok()) << "rolled back key can be reserved again";

  IdMap map2;
  map2.reserve_capacity(3);
  static_cast<void>(map2.commit(map2.reserve(9, false).value(), 0));
  auto up = map2.reserve(9, true);
  map2.rollback(up.value());
  EXPECT_EQ(map2.find(9), 0U);
  EXPECT_TRUE(map2.reserve(9, true).ok());
}

TEST(IdMap, RollbackAppendedLabelsRowButRestoresMapping) {
  IdMap map;
  map.reserve_capacity(3);
  static_cast<void>(map.commit(map.reserve(3, false).value(), 0));
  auto r = map.reserve(4, false);
  map.rollback_appended(r.value(), 1);
  EXPECT_FALSE(map.contains(4));
  EXPECT_EQ(map.label_count(), 2U);
  EXPECT_EQ(map.label(1), 4U);
  EXPECT_EQ(map.size(), 1U);
}

TEST(IdMap, Erase) {
  IdMap map;
  map.reserve_capacity(2);
  static_cast<void>(map.commit(map.reserve(11, false).value(), 0));
  const auto erased = map.erase(11);
  ASSERT_TRUE(erased.ok());
  EXPECT_EQ(erased.value(), 0U);
  EXPECT_FALSE(map.contains(11));
  EXPECT_EQ(map.size(), 0U);
  EXPECT_EQ(map.label(0), 11U);
  EXPECT_EQ(map.erase(11).status().code(), ErrorCode::NotFound);
  EXPECT_EQ(map.erase(12).status().code(), ErrorCode::NotFound);

  auto pending = map.reserve(13, false);
  ASSERT_TRUE(pending.ok());
  EXPECT_EQ(map.erase(13).status().code(), ErrorCode::NotFound)
      << "pending new key is not erasable";
}

TEST(IdMap, MemoryEstimatesGrow) {
  IdMap map;
  const auto before = map.labels_bytes() + map.map_bytes_estimate();
  map.reserve_capacity(1000);
  for (vf::InternalId i = 0; i < 1000; ++i) {
    static_cast<void>(map.commit(map.reserve(i * 7ULL, false).value(), i));
  }
  EXPECT_GT(map.labels_bytes() + map.map_bytes_estimate(), before);
  EXPECT_GE(map.labels_bytes(), 1000 * sizeof(vf::ExternalId));
}

TEST(TombstoneSet, SetTestCount) {
  TombstoneSet t;
  EXPECT_FALSE(t.test(0));
  EXPECT_FALSE(t.test(1000)) << "out of range reads as not deleted";
  t.ensure_size(130);
  EXPECT_FALSE(t.any());
  for (const vf::InternalId id : {0U, 63U, 64U, 129U}) {
    EXPECT_TRUE(t.set(id));
    EXPECT_TRUE(t.test(id));
  }
  EXPECT_FALSE(t.set(64)) << "second set reports already marked";
  EXPECT_EQ(t.count(), 4U);
  EXPECT_TRUE(t.any());
  EXPECT_FALSE(t.test(1));
  EXPECT_FALSE(t.test(65));
  t.ensure_size(10);  // never shrinks
  EXPECT_TRUE(t.test(129));
  EXPECT_GE(t.bytes(), 3 * sizeof(std::uint64_t));
}

}  // namespace
