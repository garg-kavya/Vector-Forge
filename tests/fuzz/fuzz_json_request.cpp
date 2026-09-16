// Fuzz target for the HTTP request decoders (docs/DESIGN.md §13.5, §15.1). Input: one selector
// byte (which decoder), one dimension byte, then the request body. Decoding must never crash; a
// decoded request must be internally consistent, and applying it to a collection must not crash.
//
// Built as a libFuzzer binary with -DVF_BUILD_FUZZERS=ON, and as a corpus replay test everywhere.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <string_view>
#include <vector>

#include <vectorforge/collection.hpp>

#include "server/json_codec.hpp"

namespace {

void check(bool condition) {
  if (!condition) {
    std::abort();
  }
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  if (size < 2) {
    return 0;
  }
  const std::uint8_t selector = data[0] % 5;
  const std::uint32_t dim = (data[1] % 8U) + 1U;
  const std::string_view body(reinterpret_cast<const char*>(data + 2), size - 2);
  vf::server::Limits limits;
  limits.max_batch = 64;
  limits.max_dim = 64;

  switch (selector) {
    case 0: {
      const auto req = vf::server::decode_create(body, limits);
      if (req.ok) {
        check(req.value.config.validate().ok());
        check(vf::Catalog::is_valid_name(req.value.name));
        check(req.value.config.dim <= limits.max_dim);
      } else {
        check(req.error.http_status >= 400 && !req.error.code.empty());
      }
      break;
    }
    case 1:
    case 4: {
      const auto req = selector == 1 ? vf::server::decode_insert(body, dim, limits)
                                     : vf::server::decode_bulk(body, dim, limits);
      if (!req.ok) {
        check(req.error.http_status >= 400);
        break;
      }
      check(req.value.ids.size() <= limits.max_batch);
      check(req.value.rows.size() == req.value.ids.size() * dim);
      vf::CollectionConfig cfg;
      cfg.dim = dim;
      cfg.index = vf::IndexType::Flat;
      auto c = vf::Collection::create(cfg).value();
      static_cast<void>(c->add_batch(req.value.ids, req.value.rows, {.upsert = req.value.upsert}));
      break;
    }
    default: {
      const bool batch = selector == 3;
      const auto req = vf::server::decode_search(body, dim, batch, limits);
      if (!req.ok) {
        check(req.error.http_status >= 400);
        break;
      }
      check(req.value.vectors.size() == req.value.count * dim);
      check(batch || req.value.count == 1);
      check(req.value.params.k >= 1 && req.value.params.k <= limits.max_k);
      vf::CollectionConfig cfg;
      cfg.dim = dim;
      cfg.index = vf::IndexType::Flat;
      auto c = vf::Collection::create(cfg).value();
      static_cast<void>(c->add(1, std::vector<float>(dim, 0.5F)));
      if (req.value.count > 0) {
        static_cast<void>(
            c->search(std::span<const float>(req.value.vectors).first(dim), req.value.params));
      }
      break;
    }
  }
  return 0;
}
