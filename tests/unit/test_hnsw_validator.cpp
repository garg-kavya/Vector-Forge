#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "index/hnsw/hnsw_graph.hpp"
#include "index/hnsw/hnsw_validator.hpp"

namespace {

using vf::ErrorCode;
using vf::InternalId;
using vf::detail::HnswGraph;
using vf::detail::HnswValidator;

// Node levels {1, 0, 0, 1, 0}; entry = node 0 at level 1; a valid, fully connected graph:
//   level 1: 0 <-> 3
//   level 0: 0 -> 1, 1 -> 2, 2 -> 0, 3 -> 4, 4 -> 3, 0 -> 3
HnswGraph valid_graph() {
  HnswGraph g = HnswGraph::create({.m = 2, .max_level = 3}).value();
  for (const std::uint8_t level : std::vector<std::uint8_t>{1, 0, 0, 1, 0}) {
    EXPECT_TRUE(g.add_node(level).ok());
  }
  g.set_links(0, 1, std::vector<InternalId>{3});
  g.set_links(3, 1, std::vector<InternalId>{0});
  g.set_links(0, 0, std::vector<InternalId>{1, 3});
  g.set_links(1, 0, std::vector<InternalId>{2});
  g.set_links(2, 0, std::vector<InternalId>{0});
  g.set_links(3, 0, std::vector<InternalId>{4});
  g.set_links(4, 0, std::vector<InternalId>{3});
  g.set_entry({.id = 0, .level = 1});
  return g;
}

TEST(HnswValidator, AcceptsValidGraph) {
  const HnswGraph g = valid_graph();
  const HnswValidator v(g);
  const vf::Status st = v.check_invariants();
  EXPECT_TRUE(st.ok()) << st.to_string();
  const vf::detail::HnswReachability r = v.reachability();
  EXPECT_EQ(r.nodes, 5U);
  ASSERT_EQ(r.unreachable.size(), 2U);
  EXPECT_EQ(r.unreachable[0], 0U);
  EXPECT_EQ(r.unreachable[1], 0U);
  const std::vector<std::uint64_t> hist = v.level_histogram();
  ASSERT_EQ(hist.size(), 4U);
  EXPECT_EQ(hist[0], 3U);
  EXPECT_EQ(hist[1], 2U);
  EXPECT_EQ(hist[2], 0U);
}

TEST(HnswValidator, EmptyGraph) {
  HnswGraph g = HnswGraph::create({.m = 2}).value();
  EXPECT_TRUE(HnswValidator(g).check_invariants().ok());
  EXPECT_TRUE(HnswValidator(g).reachability().unreachable.empty());
  g.set_entry({.id = 0, .level = 0});
  EXPECT_EQ(HnswValidator(g).check_invariants().code(), ErrorCode::Internal);
}

TEST(HnswValidator, DetectsSelfLink) {
  HnswGraph g = valid_graph();
  g.set_links(1, 0, std::vector<InternalId>{2, 1});
  EXPECT_EQ(HnswValidator(g).check_invariants().code(), ErrorCode::Internal);
}

TEST(HnswValidator, DetectsDuplicateLink) {
  HnswGraph g = valid_graph();
  g.set_links(1, 0, std::vector<InternalId>{2, 2});
  const vf::Status st = HnswValidator(g).check_invariants();
  EXPECT_EQ(st.code(), ErrorCode::Internal);
  EXPECT_NE(st.message().find("duplicate"), std::string::npos) << st.message();
}

TEST(HnswValidator, DetectsDanglingId) {
  HnswGraph g = valid_graph();
  g.set_links(1, 0, std::vector<InternalId>{7});
  EXPECT_EQ(HnswValidator(g).check_invariants().code(), ErrorCode::Internal);
}

TEST(HnswValidator, DetectsLinkToLowerLevelNode) {
  HnswGraph g = valid_graph();
  g.set_links(0, 1, std::vector<InternalId>{2});  // node 2 has level 0
  EXPECT_EQ(HnswValidator(g).check_invariants().code(), ErrorCode::Internal);
}

TEST(HnswValidator, DetectsBadEntryPoint) {
  HnswGraph g = valid_graph();
  g.set_entry({.id = 1, .level = 0});  // consistent with node 1, but not the top level
  EXPECT_EQ(HnswValidator(g).check_invariants().code(), ErrorCode::Internal);
  g.set_entry({.id = 1, .level = 1});  // level does not match node 1
  EXPECT_EQ(HnswValidator(g).check_invariants().code(), ErrorCode::Internal);
  g.set_entry({});
  EXPECT_EQ(HnswValidator(g).check_invariants().code(), ErrorCode::Internal);
}

TEST(HnswValidator, ReportsUnreachableNodesPerLevel) {
  HnswGraph g = valid_graph();
  g.set_links(0, 0, std::vector<InternalId>{1});  // drop 0 -> 3: nodes 3 and 4 unreachable on 0
  g.set_links(0, 1, std::vector<InternalId>{});   // and node 3 unreachable on level 1
  ASSERT_TRUE(HnswValidator(g).check_invariants().ok());
  const vf::detail::HnswReachability r = HnswValidator(g).reachability();
  ASSERT_EQ(r.unreachable.size(), 2U);
  EXPECT_EQ(r.unreachable[0], 2U);
  EXPECT_EQ(r.unreachable[1], 1U);
  EXPECT_EQ(r.unreachable_level0(), 2U);
}

}  // namespace
