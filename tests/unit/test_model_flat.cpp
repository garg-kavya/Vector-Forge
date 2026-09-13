// Model-based test: a Flat Collection and a trivially correct model receive the same seeded random
// sequence of 10^4 operations; every observable result must agree (docs/DESIGN.md §15.2).
//
// Exact configurations use small integer vectors (float arithmetic exact, many ties) and require
// identical ids, order and distances. Normalised configurations use real-valued vectors and a
// tie-tolerant comparison.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <vectorforge/collection.hpp>

#include "core/rng.hpp"
#include "support/brute_force_reference.hpp"
#include "support/test_data.hpp"

namespace {

using vf::Collection;
using vf::ErrorCode;
using vf::ExternalId;
using vf::Metric;
using vf::Neighbor;

struct ModelConfig {
  Metric metric;
  bool normalize;
  bool integer_data;
};

struct ModelEntry {
  std::vector<float> raw;
  std::uint64_t order = 0;
};

class FlatModelTest : public testing::TestWithParam<ModelConfig> {};

TEST_P(FlatModelTest, TenThousandRandomOperations) {
  const ModelConfig mc = GetParam();
  constexpr std::uint32_t kDim = 16;
  constexpr int kOperations = 10000;
  constexpr std::uint64_t kIdSpace = 400;
  const bool normalized = mc.normalize || mc.metric == Metric::Cosine;

  vf::CollectionConfig cfg;
  cfg.dim = kDim;
  cfg.metric = mc.metric;
  cfg.normalize = mc.normalize;
  cfg.index = vf::IndexType::Flat;
  const std::unique_ptr<Collection> col = Collection::create(cfg).value();

  std::map<ExternalId, ModelEntry> model;
  std::uint64_t rows = 0;  // next internal id == insertion order
  vf::detail::Xoshiro256ss rng(0xC0FFEE + static_cast<std::uint64_t>(mc.metric) * 31 +
                               (mc.normalize ? 7U : 0U));

  auto make_vector = [&]() {
    return mc.integer_data ? vf::test::integer_vector(rng, kDim)
                           : vf::test::random_vector(rng, kDim, -1.0F, 1.0F);
  };
  auto ranking_for = [&](std::span<const float> query) {
    std::vector<vf::test::RefItem> items;
    items.reserve(model.size());
    for (const auto& [id, entry] : model) {
      items.push_back({id, entry.order, entry.raw});
    }
    return vf::test::reference_ranking(mc.metric, normalized, query, items);
  };
  auto check_topk = [&](std::span<const Neighbor> actual, std::span<const float> query,
                        std::size_t k) {
    const auto ranking = ranking_for(query);
    if (mc.integer_data) {
      vf::test::expect_exact_topk(actual, ranking, k);
    } else {
      vf::test::expect_tolerant_topk(actual, ranking, k, 1e-5);
    }
  };
  // Zero vectors cannot be normalised; the model mirrors the expected rejection.
  auto is_zero = [](const std::vector<float>& v) {
    for (float x : v) {
      if (x != 0.0F) {
        return false;
      }
    }
    return true;
  };

  for (int op = 0; op < kOperations; ++op) {
    SCOPED_TRACE(testing::Message() << "operation " << op);
    const ExternalId id = vf::detail::uniform_below(rng, kIdSpace);
    const std::uint64_t choice = vf::detail::uniform_below(rng, 100);
    const bool exists = model.contains(id);

    if (choice < 35) {  // insert (plain)
      std::vector<float> v = make_vector();
      const vf::Status st = col->add(id, v);
      if (normalized && is_zero(v)) {
        ASSERT_EQ(st.code(), ErrorCode::InvalidArgument);
      } else if (exists) {
        ASSERT_EQ(st.code(), ErrorCode::AlreadyExists);
      } else {
        ASSERT_TRUE(st.ok()) << st.to_string();
        model[id] = {std::move(v), rows++};
      }
    } else if (choice < 45) {  // upsert
      std::vector<float> v = make_vector();
      const vf::Status st = col->add(id, v, {.upsert = true});
      if (normalized && is_zero(v)) {
        ASSERT_EQ(st.code(), ErrorCode::InvalidArgument);
      } else {
        ASSERT_TRUE(st.ok()) << st.to_string();
        model[id] = {std::move(v), rows++};
      }
    } else if (choice < 60) {  // remove
      const vf::Status st = col->remove(id);
      if (exists) {
        ASSERT_TRUE(st.ok());
        model.erase(id);
      } else {
        ASSERT_EQ(st.code(), ErrorCode::NotFound);
      }
    } else if (choice < 68) {  // get
      const auto got = col->get(id);
      ASSERT_EQ(got.ok(), exists);
      if (exists) {
        const std::vector<float>& raw = model[id].raw;
        const double norm = normalized ? vf::test::ref_norm(raw) : 1.0;
        for (std::size_t j = 0; j < kDim; ++j) {
          ASSERT_NEAR(got.value()[j], static_cast<double>(raw[j]) / norm, normalized ? 1e-6 : 0.0);
        }
      }
    } else if (choice < 72) {  // add_batch (fresh ids from a disjoint range, sometimes invalid)
      const std::size_t n = 1 + vf::detail::uniform_below(rng, 5);
      std::vector<ExternalId> ids;
      std::vector<float> flat;
      bool any_existing = false;
      bool any_zero = false;
      for (std::size_t i = 0; i < n; ++i) {
        const ExternalId bid = kIdSpace + vf::detail::uniform_below(rng, 200);
        if (std::find(ids.begin(), ids.end(), bid) != ids.end()) {
          continue;  // keep batch ids unique
        }
        ids.push_back(bid);
        any_existing = any_existing || model.contains(bid);
        std::vector<float> v = make_vector();
        any_zero = any_zero || is_zero(v);
        flat.insert(flat.end(), v.begin(), v.end());
      }
      const auto added = col->add_batch(ids, flat);
      // Batch validation order: shapes/values, duplicate and existing ids, then normalisability.
      if (any_existing) {
        ASSERT_EQ(added.status().code(), ErrorCode::AlreadyExists);
      } else if (normalized && any_zero) {
        ASSERT_EQ(added.status().code(), ErrorCode::InvalidArgument);
      } else {
        ASSERT_TRUE(added.ok()) << added.status().to_string();
        ASSERT_EQ(added.value(), ids.size());
        for (std::size_t i = 0; i < ids.size(); ++i) {
          const auto row = std::span<const float>(flat).subspan(i * kDim, kDim);
          model[ids[i]] = {std::vector<float>(row.begin(), row.end()), rows++};
        }
      }
    } else if (choice < 95) {  // search
      const std::uint32_t k = 1 + static_cast<std::uint32_t>(vf::detail::uniform_below(rng, 40));
      const std::vector<float> query = make_vector();
      vf::SearchParams params;
      params.k = k;
      const auto result = col->search(query, params);
      if (normalized && is_zero(query)) {
        ASSERT_EQ(result.status().code(), ErrorCode::InvalidArgument);
      } else {
        ASSERT_TRUE(result.ok()) << result.status().to_string();
        check_topk(result.value(), query, k);
      }
    } else {  // search_batch
      constexpr std::size_t kQueries = 3;
      const std::uint32_t k = 1 + static_cast<std::uint32_t>(vf::detail::uniform_below(rng, 12));
      std::vector<float> queries;
      bool any_zero = false;
      for (std::size_t q = 0; q < kQueries; ++q) {
        std::vector<float> v = make_vector();
        any_zero = any_zero || is_zero(v);
        queries.insert(queries.end(), v.begin(), v.end());
      }
      vf::SearchParams params;
      params.k = k;
      std::vector<ExternalId> ids(kQueries * k);
      std::vector<float> dists(kQueries * k);
      std::vector<std::uint32_t> counts(kQueries);
      const vf::Status st = col->search_batch(queries, kQueries, params, ids, dists, counts);
      if (normalized && any_zero) {
        ASSERT_EQ(st.code(), ErrorCode::InvalidArgument);
      } else {
        ASSERT_TRUE(st.ok()) << st.to_string();
        for (std::size_t q = 0; q < kQueries; ++q) {
          std::vector<Neighbor> hits;
          for (std::size_t j = 0; j < counts[q]; ++j) {
            hits.push_back({ids[(q * k) + j], dists[(q * k) + j]});
          }
          check_topk(hits, std::span<const float>(queries).subspan(q * kDim, kDim), k);
        }
      }
    }

    ASSERT_EQ(col->size(), model.size());
    if (op % 500 == 0) {
      const vf::CollectionStats st = col->stats();
      ASSERT_EQ(st.live_count, model.size());
      ASSERT_EQ(st.row_count, rows);
      ASSERT_EQ(st.deleted_count, rows - model.size());
      for (ExternalId probe = 0; probe < kIdSpace + 200; ++probe) {
        ASSERT_EQ(col->contains(probe), model.contains(probe)) << "id " << probe;
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Configs, FlatModelTest,
                         testing::Values(ModelConfig{Metric::L2, false, true},
                                         ModelConfig{Metric::InnerProduct, false, true},
                                         ModelConfig{Metric::Cosine, false, false},
                                         ModelConfig{Metric::L2, true, false},
                                         ModelConfig{Metric::InnerProduct, true, false}),
                         [](const testing::TestParamInfo<ModelConfig>& param_info) {
                           std::string name(vf::to_string(param_info.param.metric));
                           name += param_info.param.normalize ? "_normalized" : "";
                           name += param_info.param.integer_data ? "_exact" : "_tolerant";
                           return name;
                         });

}  // namespace
