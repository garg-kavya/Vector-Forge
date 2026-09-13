#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "index/hnsw/hnsw_graph.hpp"

namespace {

using vf::ErrorCode;
using vf::InternalId;
using vf::detail::HnswGraph;
using vf::detail::LinkView;

std::vector<InternalId> ids_of(LinkView view) {
  return {view.ids().begin(), view.ids().end()};
}

TEST(HnswGraph, CreateValidatesOptions) {
  EXPECT_EQ(HnswGraph::create({.m = 1}).status().code(), ErrorCode::InvalidArgument);
  EXPECT_EQ(HnswGraph::create({.m = 4, .max_level = 0}).status().code(),
            ErrorCode::InvalidArgument);
  EXPECT_EQ(HnswGraph::create({.m = 4, .nodes_per_chunk = 3}).status().code(),
            ErrorCode::InvalidArgument);
  // arena chunk must hold the largest block: max_level * (1 + m) = 16 * 5 = 80 words.
  EXPECT_EQ(HnswGraph::create({.m = 4, .max_level = 16, .arena_chunk_words = 64}).status().code(),
            ErrorCode::InvalidArgument);
  EXPECT_EQ(HnswGraph::create({.m = 4, .max_level = 16, .arena_chunk_words = 100}).status().code(),
            ErrorCode::InvalidArgument);
  EXPECT_TRUE(HnswGraph::create({.m = 4, .max_level = 16, .arena_chunk_words = 128}).ok());
  EXPECT_TRUE(HnswGraph::create({}).ok());
}

TEST(HnswGraph, EmptyGraph) {
  const HnswGraph g = HnswGraph::create({.m = 4}).value();
  EXPECT_EQ(g.node_count(), 0U);
  EXPECT_FALSE(g.entry().valid());
  EXPECT_EQ(g.capacity(0), 8U);
  EXPECT_EQ(g.capacity(1), 4U);
  EXPECT_EQ(g.capacity(5), 4U);
}

TEST(HnswGraph, NodesStartWithEmptyLists) {
  HnswGraph g = HnswGraph::create({.m = 4, .max_level = 3}).value();
  ASSERT_TRUE(g.add_node(0).ok());
  ASSERT_TRUE(g.add_node(2).ok());
  EXPECT_EQ(g.add_node(4).code(), ErrorCode::InvalidArgument) << "level above max_level";
  EXPECT_EQ(g.node_count(), 2U);
  EXPECT_EQ(g.level(0), 0U);
  EXPECT_EQ(g.level(1), 2U);
  EXPECT_TRUE(g.links(0, 0).empty());
  for (std::uint8_t l = 0; l <= 2; ++l) {
    EXPECT_TRUE(g.links(1, l).empty());
  }
}

TEST(HnswGraph, SetAndAppendLinks) {
  HnswGraph g = HnswGraph::create({.m = 2, .max_level = 2}).value();
  for (int i = 0; i < 6; ++i) {
    ASSERT_TRUE(g.add_node(i == 0 ? 2 : 1).ok());
  }
  const std::vector<InternalId> four = {1, 2, 3, 4};
  g.set_links(0, 0, four);
  EXPECT_EQ(ids_of(g.links(0, 0)), four);
  EXPECT_FALSE(g.try_append_link(0, 0, 5)) << "level 0 capacity is 2M = 4";

  g.set_links(0, 1, std::vector<InternalId>{3});
  EXPECT_TRUE(g.try_append_link(0, 1, 5));
  EXPECT_FALSE(g.try_append_link(0, 1, 2)) << "upper capacity is M = 2";
  EXPECT_EQ(ids_of(g.links(0, 1)), (std::vector<InternalId>{3, 5}));
  EXPECT_TRUE(g.links(0, 2).empty()) << "levels are independent";

  g.set_links(0, 0, std::vector<InternalId>{5});  // shrinking keeps the prefix semantics
  EXPECT_EQ(ids_of(g.links(0, 0)), (std::vector<InternalId>{5}));
  g.set_links(0, 0, {});
  EXPECT_TRUE(g.links(0, 0).empty());
}

TEST(HnswGraph, EntryPoint) {
  HnswGraph g = HnswGraph::create({.m = 4}).value();
  ASSERT_TRUE(g.add_node(3).ok());
  g.set_entry({.id = 0, .level = 3});
  EXPECT_TRUE(g.entry().valid());
  EXPECT_EQ(g.entry(), (HnswGraph::Entry{.id = 0, .level = 3}));
}

TEST(HnswGraph, StableAddressesAcrossChunks) {
  // Tiny chunks: 4 nodes per level-0 chunk and an arena of 16 words (max block 2 * (1 + 3) = 8).
  HnswGraph g =
      HnswGraph::create({.m = 3, .max_level = 2, .nodes_per_chunk = 4, .arena_chunk_words = 16})
          .value();
  constexpr std::uint32_t kNodes = 50;
  std::vector<const InternalId*> l0_addresses;
  std::vector<const InternalId*> upper_addresses;
  for (std::uint32_t i = 0; i < kNodes; ++i) {
    const auto level = static_cast<std::uint8_t>(i % 3);
    ASSERT_TRUE(g.add_node(level).ok());
    // Give every list a recognisable content: level-l list of node i = {i+1+l} (mod kNodes).
    for (std::uint8_t l = 0; l <= level; ++l) {
      g.set_links(i, l, std::vector<InternalId>{(i + 1 + l) % kNodes});
    }
    l0_addresses.push_back(g.links(i, 0).ids().data());
    upper_addresses.push_back(level > 0 ? g.links(i, level).ids().data() : nullptr);
  }
  for (std::uint32_t i = 0; i < kNodes; ++i) {
    const auto level = static_cast<std::uint8_t>(i % 3);
    EXPECT_EQ(g.links(i, 0).ids().data(), l0_addresses[i]) << "node " << i;
    if (level > 0) {
      EXPECT_EQ(g.links(i, level).ids().data(), upper_addresses[i]) << "node " << i;
    }
    for (std::uint8_t l = 0; l <= level; ++l) {
      ASSERT_EQ(g.links(i, l).size(), 1U);
      EXPECT_EQ(g.links(i, l)[0], (i + 1 + l) % kNodes) << "node " << i << " level " << +l;
    }
  }
  EXPECT_GT(g.bytes(), 0U);
}

TEST(HnswGraph, CanonicalBytesReflectLogicalContentOnly) {
  auto build = [](std::size_t nodes_per_chunk, bool extra_stale_link) {
    HnswGraph g =
        HnswGraph::create({.m = 2, .max_level = 2, .nodes_per_chunk = nodes_per_chunk}).value();
    for (int i = 0; i < 5; ++i) {
      EXPECT_TRUE(g.add_node(i == 2 ? 1 : 0).ok());
    }
    if (extra_stale_link) {
      g.set_links(0, 0, std::vector<InternalId>{4, 3, 2});
    }
    g.set_links(0, 0, std::vector<InternalId>{1});  // stale ids past the count must not matter
    g.set_links(2, 1, std::vector<InternalId>{});
    g.set_links(2, 0, std::vector<InternalId>{0, 1});
    g.set_entry({.id = 2, .level = 1});
    return g.canonical_bytes();
  };
  const std::vector<std::uint8_t> a = build(0, false);
  EXPECT_EQ(a, build(2, false)) << "chunking is not part of the logical graph";
  EXPECT_EQ(a, build(0, true)) << "stale slots are not part of the logical graph";

  HnswGraph other = HnswGraph::create({.m = 2, .max_level = 2}).value();
  for (int i = 0; i < 5; ++i) {
    ASSERT_TRUE(other.add_node(i == 2 ? 1 : 0).ok());
  }
  other.set_links(0, 0, std::vector<InternalId>{1});
  other.set_links(2, 0, std::vector<InternalId>{1, 0});  // same set, different order
  other.set_entry({.id = 2, .level = 1});
  EXPECT_NE(a, other.canonical_bytes());
}

}  // namespace
