#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>

#include <vectorforge/config.hpp>
#include <vectorforge/types.hpp>

namespace {

using vf::CollectionConfig;
using vf::ErrorCode;
using vf::HnswParams;
using vf::IndexType;
using vf::Metric;

TEST(Types, Constants) {
  EXPECT_EQ(vf::kInvalidInternalId, 0xFFFFFFFFU);
  EXPECT_EQ(vf::kMaxVectorsPerCollection, 0xFFFFFFFFULL);
  EXPECT_EQ(vf::kMaxDim, 65536U);
  // Overflow argument for kMaxAbsComponent: (2*max)^2 * kMaxDim must stay below FLT_MAX.
  const double worst = std::pow(2.0 * static_cast<double>(vf::kMaxAbsComponent), 2.0) * vf::kMaxDim;
  EXPECT_LT(worst, 3.4028234663852886e38);
}

TEST(Types, MetricRoundTrip) {
  for (const Metric m : {Metric::L2, Metric::InnerProduct, Metric::Cosine}) {
    EXPECT_TRUE(vf::is_valid(m));
    const auto parsed = vf::parse_metric(vf::to_string(m));
    ASSERT_TRUE(parsed.ok()) << parsed.status().to_string();
    EXPECT_EQ(parsed.value(), m);
  }
  EXPECT_EQ(vf::parse_metric("inner_product").value(), Metric::InnerProduct);
  EXPECT_EQ(vf::parse_metric("L2").status().code(), ErrorCode::InvalidArgument);
  EXPECT_EQ(vf::parse_metric("").status().code(), ErrorCode::InvalidArgument);
  EXPECT_FALSE(vf::is_valid(static_cast<Metric>(7)));
  EXPECT_EQ(vf::to_string(static_cast<Metric>(7)), "invalid");
}

TEST(Types, IndexTypeRoundTrip) {
  for (const IndexType t : {IndexType::Flat, IndexType::Hnsw}) {
    EXPECT_TRUE(vf::is_valid(t));
    EXPECT_EQ(vf::parse_index_type(vf::to_string(t)).value(), t);
  }
  EXPECT_EQ(vf::parse_index_type("ivf").status().code(), ErrorCode::InvalidArgument);
  EXPECT_FALSE(vf::is_valid(static_cast<IndexType>(9)));
}

TEST(Types, NeighborEquality) {
  EXPECT_EQ((vf::Neighbor{1, 0.5F}), (vf::Neighbor{1, 0.5F}));
  EXPECT_FALSE((vf::Neighbor{1, 0.5F}) == (vf::Neighbor{2, 0.5F}));
}

TEST(HnswParams, DefaultsAreValid) {
  const HnswParams p;
  EXPECT_TRUE(p.validate().ok()) << p.validate().to_string();
  EXPECT_EQ(p.M, 16U);
  EXPECT_EQ(p.max_links_level0(), 32U);
  EXPECT_EQ(p.max_links_upper(), 16U);
  EXPECT_NEAR(p.level_multiplier(), 1.0 / std::log(16.0), 1e-15);
}

TEST(HnswParams, RejectsOutOfRange) {
  auto expect_invalid = [](HnswParams p) {
    EXPECT_EQ(p.validate().code(), ErrorCode::InvalidArgument);
  };
  HnswParams p;
  p.M = 1;
  expect_invalid(p);
  p = {};
  p.M = HnswParams::kMaxM + 1;
  expect_invalid(p);
  p = {};
  p.ef_construction = p.M - 1;
  expect_invalid(p);
  p = {};
  p.ef_construction = HnswParams::kMaxEf + 1;
  expect_invalid(p);
  p = {};
  p.ef_search = 0;
  expect_invalid(p);
  p = {};
  p.max_level = 0;
  expect_invalid(p);
  p = {};
  p.max_level = HnswParams::kMaxLevelCap + 1;
  expect_invalid(p);
}

TEST(HnswParams, AcceptsBoundaries) {
  HnswParams p;
  p.M = HnswParams::kMinM;
  p.ef_construction = HnswParams::kMinM;
  p.ef_search = 1;
  p.max_level = 1;
  EXPECT_TRUE(p.validate().ok());
  p.M = HnswParams::kMaxM;
  p.ef_construction = HnswParams::kMaxEf;
  p.ef_search = HnswParams::kMaxEf;
  p.max_level = HnswParams::kMaxLevelCap;
  EXPECT_TRUE(p.validate().ok());
}

TEST(CollectionConfig, Validation) {
  CollectionConfig c;
  EXPECT_EQ(c.validate().code(), ErrorCode::InvalidArgument);  // dim = 0
  c.dim = 128;
  EXPECT_TRUE(c.validate().ok());
  c.dim = vf::kMaxDim;
  EXPECT_TRUE(c.validate().ok());
  c.dim = vf::kMaxDim + 1;
  EXPECT_EQ(c.validate().code(), ErrorCode::InvalidArgument);

  c.dim = 8;
  c.metric = static_cast<Metric>(42);
  EXPECT_EQ(c.validate().code(), ErrorCode::InvalidArgument);
  c.metric = Metric::L2;
  c.index = static_cast<IndexType>(42);
  EXPECT_EQ(c.validate().code(), ErrorCode::InvalidArgument);

  // HNSW params are validated only for HNSW collections.
  c.index = IndexType::Hnsw;
  c.hnsw.M = 0;
  EXPECT_EQ(c.validate().code(), ErrorCode::InvalidArgument);
  c.index = IndexType::Flat;
  EXPECT_TRUE(c.validate().ok());
}

TEST(CollectionConfig, EffectiveNormalize) {
  CollectionConfig c;
  c.metric = Metric::Cosine;
  EXPECT_TRUE(c.effective_normalize());
  c.metric = Metric::L2;
  EXPECT_FALSE(c.effective_normalize());
  c.normalize = true;
  EXPECT_TRUE(c.effective_normalize());
}

}  // namespace
