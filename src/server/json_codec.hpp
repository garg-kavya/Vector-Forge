#pragma once

// Request decoding and response encoding for the HTTP API (docs/http-api.md).
//
// Decoders are strict: the body must be a JSON object of the documented shape, unknown keys are
// rejected, ids are unsigned integers, vector components are numbers. Vector values themselves
// (finite, in range, dimension) are validated again by the Collection. Every decoder is safe on
// arbitrary input (fuzzed by tests/fuzz/fuzz_json_request.cpp).

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <vectorforge/catalog.hpp>
#include <vectorforge/collection.hpp>
#include <vectorforge/status.hpp>

#include "server/limits.hpp"

namespace vf::server {

// An error with the HTTP status it maps to (docs/DESIGN.md §13.4).
struct ApiError {
  int http_status = 500;
  std::string code;  // "DIMENSION_MISMATCH", "MALFORMED_JSON", ...
  std::string message;
};

[[nodiscard]] ApiError to_api_error(const Status& status);
[[nodiscard]] std::string encode_error(const ApiError& error);

struct CreateRequest {
  std::string name;
  CollectionConfig config;
};

struct InsertRequest {
  std::vector<ExternalId> ids;
  std::vector<float> rows;  // ids.size() * dim, row-major
  bool upsert = false;
};

struct SearchRequest {
  std::vector<float> vectors;  // nq * dim
  std::size_t count = 0;       // nq
  SearchParams params;
};

// A decoding result: the value, or an ApiError.
template <class T>
struct Decoded {
  T value{};
  bool ok = false;
  ApiError error;
};

[[nodiscard]] Decoded<CreateRequest> decode_create(std::string_view body, const Limits& limits);
[[nodiscard]] Decoded<InsertRequest> decode_insert(std::string_view body, std::uint32_t dim,
                                                   const Limits& limits);
// Binary bulk body (docs/http-api.md, "vectors:bulk").
[[nodiscard]] Decoded<InsertRequest> decode_bulk(std::string_view body, std::uint32_t dim,
                                                 const Limits& limits);
// Single query: {"vector": [...], "k": 10, "ef_search": 64}; batch: {"vectors": [[...], ...], ...}.
[[nodiscard]] Decoded<SearchRequest> decode_search(std::string_view body, std::uint32_t dim,
                                                   bool batch, const Limits& limits);

[[nodiscard]] std::string encode_config(const CollectionConfig& config);  // JSON object
[[nodiscard]] std::string encode_collection(std::string_view name, const Collection& collection);
[[nodiscard]] std::string encode_stats(std::string_view name, const Collection& collection);
[[nodiscard]] std::string encode_neighbors(std::span<const Neighbor> neighbors);
[[nodiscard]] std::string encode_vector(ExternalId id, std::span<const float> vector);
[[nodiscard]] std::string format_ms(double milliseconds);

// Maximum nesting depth of accepted JSON bodies (checked before parsing, so deeply nested input
// cannot exhaust the stack of the recursive parser).
inline constexpr std::size_t kMaxJsonDepth = 8;
[[nodiscard]] bool within_depth(std::string_view body, std::size_t max_depth) noexcept;

}  // namespace vf::server
