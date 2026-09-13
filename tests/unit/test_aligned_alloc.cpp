#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

#include "core/aligned_alloc.hpp"

namespace {

using vf::detail::AlignedAllocator;
using vf::detail::make_aligned_array;
using vf::detail::make_aligned_array_zeroed;

bool is_aligned(const void* ptr, std::size_t alignment) {
  return (reinterpret_cast<std::uintptr_t>(ptr) % alignment) == 0;
}

TEST(AlignedAlloc, DefaultIsCacheLineAligned) {
  EXPECT_EQ(vf::detail::kCacheLineAlignment, 64U);
  for (std::size_t count : {1U, 3U, 17U, 1000U, 65537U}) {
    const auto array = make_aligned_array<float>(count);
    ASSERT_NE(array, nullptr);
    EXPECT_TRUE(is_aligned(array.get(), 64)) << "count=" << count;
    array[count - 1] = 1.0F;  // writable end-to-end (ASan verifies bounds)
    array[0] = 2.0F;
  }
}

TEST(AlignedAlloc, CustomAlignments) {
  for (int i = 0; i < 16; ++i) {
    EXPECT_TRUE(is_aligned(make_aligned_array<double, 16>(5).get(), 16));
    EXPECT_TRUE(is_aligned(make_aligned_array<std::uint32_t, 32>(5).get(), 32));
    EXPECT_TRUE(is_aligned(make_aligned_array<std::uint8_t, 4096>(5).get(), 4096));
  }
}

TEST(AlignedAlloc, ZeroCountIsNull) {
  EXPECT_EQ(make_aligned_array<float>(0), nullptr);
}

TEST(AlignedAlloc, ZeroedContents) {
  const auto array = make_aligned_array_zeroed<float>(1025);
  for (std::size_t i = 0; i < 1025; ++i) {
    ASSERT_EQ(array[i], 0.0F);
  }
}

TEST(AlignedAlloc, OverflowingCountThrows) {
  const std::size_t huge = std::numeric_limits<std::size_t>::max() / 2;
  EXPECT_THROW((void)make_aligned_array<float>(huge), std::bad_array_new_length);
}

TEST(AlignedAllocator, VectorStorageIsAligned) {
  std::vector<float, AlignedAllocator<float>> v;
  for (int i = 0; i < 10000; ++i) {
    v.push_back(static_cast<float>(i));
    ASSERT_TRUE(is_aligned(v.data(), 64));
  }
  EXPECT_EQ(v[9999], 9999.0F);

  std::vector<float, AlignedAllocator<float, 32>> w(123, 1.5F);
  EXPECT_TRUE(is_aligned(w.data(), 32));
  EXPECT_EQ(w[122], 1.5F);
}

TEST(AlignedAllocator, RebindAndEquality) {
  using FloatAlloc = AlignedAllocator<float, 64>;
  using ByteAlloc = std::allocator_traits<FloatAlloc>::rebind_alloc<std::uint8_t>;
  const FloatAlloc a;
  const ByteAlloc b(a);
  EXPECT_TRUE(FloatAlloc(b) == a);
  ByteAlloc copy = b;
  std::uint8_t* p = copy.allocate(3);
  EXPECT_TRUE(is_aligned(p, 64));
  copy.deallocate(p, 3);
}

}  // namespace
