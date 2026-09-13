#include <gtest/gtest.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <vectorforge/types.hpp>

#include "storage/vector_store.hpp"

namespace {

using vf::ErrorCode;
using vf::InternalId;
using vf::detail::VectorStore;

std::vector<float> row_for(std::uint64_t id, std::uint32_t dim) {
  std::vector<float> row(dim);
  for (std::uint32_t j = 0; j < dim; ++j) {
    row[j] = static_cast<float>(id) * 1000.0F + static_cast<float>(j);
  }
  return row;
}

VectorStore make_store(std::uint32_t dim, std::size_t rows_per_chunk,
                       std::uint64_t max_rows = vf::kMaxVectorsPerCollection) {
  auto store =
      VectorStore::create({.dim = dim, .rows_per_chunk = rows_per_chunk, .max_rows = max_rows});
  EXPECT_TRUE(store.ok()) << store.status().to_string();
  return std::move(store).value();
}

TEST(VectorStore, CreateValidation) {
  EXPECT_EQ(VectorStore::create({.dim = 0}).status().code(), ErrorCode::InvalidArgument);
  EXPECT_EQ(VectorStore::create({.dim = vf::kMaxDim + 1}).status().code(),
            ErrorCode::InvalidArgument);
  EXPECT_EQ(VectorStore::create({.dim = 4, .rows_per_chunk = 3}).status().code(),
            ErrorCode::InvalidArgument);
  EXPECT_EQ(VectorStore::create({.dim = 4, .rows_per_chunk = 4, .max_rows = 0}).status().code(),
            ErrorCode::InvalidArgument);
  EXPECT_EQ(VectorStore::create(
                {.dim = 4, .rows_per_chunk = 4, .max_rows = vf::kMaxVectorsPerCollection + 1})
                .status()
                .code(),
            ErrorCode::InvalidArgument);
  EXPECT_TRUE(VectorStore::create({.dim = vf::kMaxDim}).ok());
}

TEST(VectorStore, DefaultRowsPerChunk) {
  EXPECT_EQ(VectorStore::default_rows_per_chunk(1), VectorStore::kMaxDefaultRowsPerChunk);
  EXPECT_EQ(VectorStore::default_rows_per_chunk(128), 32768U);  // 16 MiB / 512 B
  EXPECT_EQ(VectorStore::default_rows_per_chunk(768), 4096U);   // floor2(5461)
  EXPECT_EQ(VectorStore::default_rows_per_chunk(1536), 2048U);  // floor2(2730)
  EXPECT_EQ(VectorStore::default_rows_per_chunk(vf::kMaxDim), 64U);
  for (std::uint32_t d = 1; d <= 70000; d += 997) {
    const std::size_t rows = VectorStore::default_rows_per_chunk(d);
    EXPECT_TRUE(std::has_single_bit(rows));
    EXPECT_GE(rows, 1U);
  }
}

TEST(VectorStore, EmptyStore) {
  const VectorStore store = make_store(3, 4);
  EXPECT_TRUE(store.empty());
  EXPECT_EQ(store.size(), 0U);
  EXPECT_EQ(store.capacity(), 0U);
  EXPECT_EQ(store.chunk_count(), 0U);
  EXPECT_EQ(store.allocated_bytes(), 0U);
  EXPECT_EQ(store.dim(), 3U);
}

TEST(VectorStore, AppendAcrossChunkBoundaries) {
  constexpr std::uint32_t kDim = 5;
  VectorStore store = make_store(kDim, 4);
  for (std::uint64_t i = 0; i < 13; ++i) {
    const auto id = store.append(row_for(i, kDim));
    ASSERT_TRUE(id.ok());
    EXPECT_EQ(id.value(), static_cast<InternalId>(i));
    EXPECT_EQ(store.size(), i + 1);
    EXPECT_EQ(store.chunk_count(), (i / 4) + 1);  // boundaries at 3|4, 7|8, 11|12
  }
  EXPECT_EQ(store.capacity(), 16U);
  EXPECT_EQ(store.chunk_bytes(), 4U * kDim * sizeof(float));
  EXPECT_EQ(store.allocated_bytes(), 4 * store.chunk_bytes());
  for (std::uint64_t i = 0; i < 13; ++i) {
    const auto row = store.row(static_cast<InternalId>(i));
    ASSERT_EQ(row.size(), kDim);
    EXPECT_EQ(std::vector<float>(row.begin(), row.end()), row_for(i, kDim)) << "row " << i;
  }
}

TEST(VectorStore, AddressesAreStableAndAligned) {
  constexpr std::uint32_t kDim = 16;
  VectorStore store = make_store(kDim, 8);
  ASSERT_TRUE(store.append(row_for(0, kDim)).ok());
  ASSERT_TRUE(store.append(row_for(1, kDim)).ok());
  const float* p0 = store.row_ptr(0);
  const float* p1 = store.row_ptr(1);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p0) % VectorStore::kAlignment, 0U);

  for (std::uint64_t i = 2; i < 1000; ++i) {
    ASSERT_TRUE(store.append(row_for(i, kDim)).ok());
  }
  EXPECT_EQ(store.row_ptr(0), p0);
  EXPECT_EQ(store.row_ptr(1), p1);
  EXPECT_EQ(p0[kDim - 1], row_for(0, kDim)[kDim - 1]);
  for (std::uint64_t i = 0; i < 1000; i += 8) {
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(store.row_ptr(static_cast<InternalId>(i))) %
                  VectorStore::kAlignment,
              0U);
  }
}

TEST(VectorStore, AppendErrors) {
  VectorStore store = make_store(3, 2, /*max_rows=*/3);
  EXPECT_EQ(store.append(std::vector<float>{1.0F, 2.0F}).status().code(),
            ErrorCode::DimensionMismatch);
  EXPECT_EQ(store.size(), 0U);
  for (int i = 0; i < 3; ++i) {
    ASSERT_TRUE(store.append(row_for(static_cast<std::uint64_t>(i), 3)).ok());
  }
  EXPECT_EQ(store.append(row_for(3, 3)).status().code(), ErrorCode::ResourceExhausted);
  EXPECT_EQ(store.size(), 3U);
}

TEST(VectorStore, Reserve) {
  VectorStore store = make_store(2, 4, /*max_rows=*/10);
  ASSERT_TRUE(store.reserve(9).ok());
  EXPECT_EQ(store.capacity(), 12U);
  EXPECT_EQ(store.chunk_count(), 3U);
  EXPECT_EQ(store.size(), 0U);
  ASSERT_TRUE(store.reserve(1).ok());  // no shrink
  EXPECT_EQ(store.chunk_count(), 3U);
  EXPECT_EQ(store.reserve(11).code(), ErrorCode::ResourceExhausted);
  for (int i = 0; i < 10; ++i) {
    ASSERT_TRUE(store.append(row_for(static_cast<std::uint64_t>(i), 2)).ok());
  }
  EXPECT_EQ(store.chunk_count(), 3U);
}

TEST(VectorStore, PopBackUndoesLastAppend) {
  constexpr std::uint32_t kDim = 3;
  VectorStore store = make_store(kDim, 2);
  for (std::uint64_t i = 0; i < 3; ++i) {
    ASSERT_TRUE(store.append(row_for(i, kDim)).ok());
  }
  const float* p2 = store.row_ptr(2);
  store.pop_back();  // removes the only row of the second chunk; the chunk is kept
  EXPECT_EQ(store.size(), 2U);
  EXPECT_EQ(store.chunk_count(), 2U);
  const auto id = store.append(row_for(9, kDim));
  ASSERT_TRUE(id.ok());
  EXPECT_EQ(id.value(), 2U) << "the next append reuses the id";
  EXPECT_EQ(store.row_ptr(2), p2) << "and the same storage";
  EXPECT_EQ(std::vector<float>(store.row(2).begin(), store.row(2).end()), row_for(9, kDim));
  EXPECT_EQ(std::vector<float>(store.row(1).begin(), store.row(1).end()), row_for(1, kDim));
}

TEST(VectorStore, MutableRow) {
  VectorStore store = make_store(3, 4);
  ASSERT_TRUE(store.append(std::vector<float>{1.0F, 2.0F, 3.0F}).ok());
  store.mutable_row_ptr(0)[1] = 42.0F;
  EXPECT_EQ(store.row(0)[1], 42.0F);
}

TEST(VectorStore, MoveTransfersOwnership) {
  VectorStore a = make_store(4, 2);
  for (int i = 0; i < 5; ++i) {
    ASSERT_TRUE(a.append(row_for(static_cast<std::uint64_t>(i), 4)).ok());
  }
  const float* p = a.row_ptr(4);

  VectorStore b = std::move(a);
  EXPECT_EQ(b.size(), 5U);
  EXPECT_EQ(b.row_ptr(4), p);      // no reallocation on move
  EXPECT_EQ(a.size(), 0U);         // NOLINT(bugprone-use-after-move): moved-from state is specified
  EXPECT_EQ(a.chunk_count(), 0U);  // NOLINT(bugprone-use-after-move)

  VectorStore c = make_store(4, 2);
  c = std::move(b);
  EXPECT_EQ(c.size(), 5U);
  EXPECT_EQ(std::vector<float>(c.row(3).begin(), c.row(3).end()), row_for(3, 4));
}

}  // namespace
