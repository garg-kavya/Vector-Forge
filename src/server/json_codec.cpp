#include "server/json_codec.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <set>
#include <utility>

#include <nlohmann/json.hpp>

namespace vf::server {

namespace {

using json = nlohmann::json;

ApiError make_error(int status, std::string code, std::string message) {
  return {.http_status = status, .code = std::move(code), .message = std::move(message)};
}

template <class T>
Decoded<T> fail(int status, std::string code, std::string message) {
  Decoded<T> out;
  out.error = make_error(status, std::move(code), std::move(message));
  return out;
}

template <class T>
Decoded<T> bad_request(std::string message) {
  return fail<T>(400, "INVALID_ARGUMENT", std::move(message));
}

template <class T>
Decoded<T> limit_exceeded(std::string message) {
  return fail<T>(422, "LIMIT_EXCEEDED", std::move(message));
}

// Parses `body` as a JSON object whose keys are all in `allowed`.
template <class T>
bool parse_object(std::string_view body, std::initializer_list<std::string_view> allowed, json& out,
                  Decoded<T>& error) {
  if (!within_depth(body, kMaxJsonDepth)) {
    error = fail<T>(400, "MALFORMED_JSON",
                    "JSON nesting deeper than " + std::to_string(kMaxJsonDepth) + " levels");
    return false;
  }
  out = json::parse(body, nullptr, false);
  if (out.is_discarded()) {
    error = fail<T>(400, "MALFORMED_JSON", "request body is not valid JSON");
    return false;
  }
  if (!out.is_object()) {
    error = bad_request<T>("request body must be a JSON object");
    return false;
  }
  for (const auto& item : out.items()) {
    bool known = false;
    for (const std::string_view key : allowed) {
      known = known || item.key() == key;
    }
    if (!known) {
      error = bad_request<T>("unknown field '" + item.key() + "'");
      return false;
    }
  }
  return true;
}

// Unsigned integer that fits `max`; JSON floats (even 1.0) are rejected.
bool get_uint(const json& value, std::uint64_t max, std::uint64_t& out) {
  if (value.is_number_unsigned()) {
    out = value.get<std::uint64_t>();
  } else if (value.is_number_integer() && value.get<std::int64_t>() >= 0) {
    out = static_cast<std::uint64_t>(value.get<std::int64_t>());
  } else {
    return false;
  }
  return out <= max;
}

// Appends the components of `value` (an array of `dim` numbers) to `out`.
template <class T>
bool append_vector(const json& value, std::uint32_t dim, const std::string& what,
                   std::vector<float>& out, Decoded<T>& error) {
  if (!value.is_array()) {
    error = bad_request<T>(what + " must be an array of numbers");
    return false;
  }
  if (value.size() != dim) {
    error = fail<T>(400, "DIMENSION_MISMATCH",
                    what + " has " + std::to_string(value.size()) + " components, expected " +
                        std::to_string(dim));
    return false;
  }
  for (const json& component : value) {
    if (!component.is_number()) {
      error = bad_request<T>(what + " must contain only numbers");
      return false;
    }
    const double d = component.get<double>();
    // Values beyond float range become infinite here and are rejected by the collection.
    constexpr float kInf = std::numeric_limits<float>::infinity();
    const bool too_large = std::abs(d) > static_cast<double>(std::numeric_limits<float>::max());
    out.push_back(too_large ? (d > 0 ? kInf : -kInf) : static_cast<float>(d));
  }
  return true;
}

template <class T>
bool read_search_params(const json& body, const Limits& limits, SearchParams& params,
                        Decoded<T>& error) {
  std::uint64_t value = 0;
  if (const auto it = body.find("k"); it != body.end()) {
    if (!get_uint(*it, std::numeric_limits<std::uint32_t>::max(), value) || value == 0) {
      error = bad_request<T>("'k' must be a positive integer");
      return false;
    }
    if (value > limits.max_k) {
      error = limit_exceeded<T>("'k' is " + std::to_string(value) + ", the limit is " +
                                std::to_string(limits.max_k));
      return false;
    }
    params.k = static_cast<std::uint32_t>(value);
  }
  if (const auto it = body.find("ef_search"); it != body.end()) {
    if (!get_uint(*it, std::numeric_limits<std::uint32_t>::max(), value) || value == 0) {
      error = bad_request<T>("'ef_search' must be a positive integer");
      return false;
    }
    if (value > limits.max_ef) {
      error = limit_exceeded<T>("'ef_search' is " + std::to_string(value) + ", the limit is " +
                                std::to_string(limits.max_ef));
      return false;
    }
    params.ef_search = static_cast<std::uint32_t>(value);
  }
  return true;
}

void read_u64(std::string_view bytes, std::size_t offset, std::uint64_t& out) {
  out = 0;
  for (std::size_t i = 0; i < 8; ++i) {
    out |= std::uint64_t{static_cast<unsigned char>(bytes[offset + i])} << (8U * i);
  }
}

}  // namespace

bool within_depth(std::string_view body, std::size_t max_depth) noexcept {
  std::size_t depth = 0;
  bool in_string = false;
  bool escaped = false;
  for (const char c : body) {
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else if (c == '"') {
        in_string = false;
      }
      continue;
    }
    if (c == '"') {
      in_string = true;
    } else if (c == '[' || c == '{') {
      if (++depth > max_depth) {
        return false;
      }
    } else if ((c == ']' || c == '}') && depth > 0) {
      --depth;
    }
  }
  return true;
}

ApiError to_api_error(const Status& status) {
  int http = 500;
  switch (status.code()) {
    case ErrorCode::Ok:
      http = 200;
      break;
    case ErrorCode::InvalidArgument:
    case ErrorCode::DimensionMismatch:
      http = 400;
      break;
    case ErrorCode::NotFound:
      http = 404;
      break;
    case ErrorCode::AlreadyExists:
    case ErrorCode::FailedPrecondition:
      http = 409;
      break;
    case ErrorCode::ResourceExhausted:
      http = 507;
      break;
    case ErrorCode::Unavailable:
      http = 503;
      break;
    case ErrorCode::CorruptData:
    case ErrorCode::UnsupportedVersion:
    case ErrorCode::IoError:
    case ErrorCode::Internal:
      http = 500;
      break;
  }
  return make_error(http, std::string(to_string(status.code())), status.message());
}

std::string encode_error(const ApiError& error) {
  json body;
  body["error"] = {{"code", error.code}, {"message", error.message}};
  return body.dump(-1, ' ', false, json::error_handler_t::replace);
}

Decoded<CreateRequest> decode_create(std::string_view body, const Limits& limits) {
  Decoded<CreateRequest> out;
  json doc;
  if (!parse_object(body, {"name", "dim", "metric", "normalize", "index"}, doc, out)) {
    return out;
  }
  CreateRequest& req = out.value;
  const auto name = doc.find("name");
  if (name == doc.end() || !name->is_string()) {
    return bad_request<CreateRequest>("'name' (string) is required");
  }
  req.name = name->get<std::string>();
  if (!Catalog::is_valid_name(req.name)) {
    return bad_request<CreateRequest>("'name' must match [A-Za-z0-9_-]{1,64}");
  }
  std::uint64_t value = 0;
  const auto dim = doc.find("dim");
  if (dim == doc.end() || !get_uint(*dim, std::numeric_limits<std::uint32_t>::max(), value) ||
      value == 0) {
    return bad_request<CreateRequest>("'dim' (positive integer) is required");
  }
  if (value > limits.max_dim) {
    return limit_exceeded<CreateRequest>("'dim' is " + std::to_string(value) + ", the limit is " +
                                         std::to_string(limits.max_dim));
  }
  req.config.dim = static_cast<std::uint32_t>(value);
  if (const auto metric = doc.find("metric"); metric != doc.end()) {
    Result<Metric> parsed =
        metric->is_string() ? parse_metric(metric->get<std::string>())
                            : Result<Metric>(Status::invalid_argument("'metric' must be a string"));
    if (!parsed.ok()) {
      return bad_request<CreateRequest>(parsed.status().message());
    }
    req.config.metric = parsed.value();
  }
  if (const auto normalize = doc.find("normalize"); normalize != doc.end()) {
    if (!normalize->is_boolean()) {
      return bad_request<CreateRequest>("'normalize' must be a boolean");
    }
    req.config.normalize = normalize->get<bool>();
  }
  if (const auto index = doc.find("index"); index != doc.end()) {
    if (!index->is_object()) {
      return bad_request<CreateRequest>("'index' must be an object");
    }
    for (const auto& item : index->items()) {
      const std::string& key = item.key();
      const json& v = item.value();
      if (key == "type") {
        Result<IndexType> type =
            v.is_string()
                ? parse_index_type(v.get<std::string>())
                : Result<IndexType>(Status::invalid_argument("'index.type' must be a string"));
        if (!type.ok()) {
          return bad_request<CreateRequest>(type.status().message());
        }
        req.config.index = type.value();
      } else if (key == "M" || key == "ef_construction" || key == "ef_search" ||
                 key == "max_level") {
        if (!get_uint(v, std::numeric_limits<std::uint32_t>::max(), value)) {
          return bad_request<CreateRequest>("'index." + key + "' must be a non-negative integer");
        }
        if (key == "M") {
          req.config.hnsw.M = static_cast<std::uint32_t>(value);
        } else if (key == "ef_construction") {
          req.config.hnsw.ef_construction = static_cast<std::uint32_t>(value);
        } else if (key == "ef_search") {
          req.config.hnsw.ef_search = static_cast<std::uint32_t>(value);
        } else if (value > HnswParams::kMaxLevelCap) {
          return bad_request<CreateRequest>("'index.max_level' must be at most " +
                                            std::to_string(HnswParams::kMaxLevelCap));
        } else {
          req.config.hnsw.max_level = static_cast<std::uint8_t>(value);
        }
      } else if (key == "seed") {
        if (!get_uint(v, std::numeric_limits<std::uint64_t>::max(), req.config.hnsw.seed)) {
          return bad_request<CreateRequest>("'index.seed' must be a non-negative integer");
        }
      } else if (key == "concurrency") {
        Result<Concurrency> mode = v.is_string() ? parse_concurrency(v.get<std::string>())
                                                 : Result<Concurrency>(Status::invalid_argument(
                                                       "'index.concurrency' must be a string"));
        if (!mode.ok()) {
          return bad_request<CreateRequest>(mode.status().message());
        }
        req.config.concurrency = mode.value();
      } else {
        return bad_request<CreateRequest>("unknown field 'index." + key + "'");
      }
    }
  }
  if (Status st = req.config.validate(); !st.ok()) {
    return bad_request<CreateRequest>(st.message());
  }
  out.ok = true;
  return out;
}

Decoded<InsertRequest> decode_insert(std::string_view body, std::uint32_t dim,
                                     const Limits& limits) {
  Decoded<InsertRequest> out;
  json doc;
  if (!parse_object(body, {"upsert", "vectors"}, doc, out)) {
    return out;
  }
  InsertRequest& req = out.value;
  if (const auto upsert = doc.find("upsert"); upsert != doc.end()) {
    if (!upsert->is_boolean()) {
      return bad_request<InsertRequest>("'upsert' must be a boolean");
    }
    req.upsert = upsert->get<bool>();
  }
  const auto vectors = doc.find("vectors");
  if (vectors == doc.end() || !vectors->is_array()) {
    return bad_request<InsertRequest>("'vectors' (array) is required");
  }
  if (vectors->size() > limits.max_batch) {
    return limit_exceeded<InsertRequest>(std::to_string(vectors->size()) +
                                         " vectors exceed the batch limit of " +
                                         std::to_string(limits.max_batch));
  }
  req.ids.reserve(vectors->size());
  req.rows.reserve(vectors->size() * dim);
  for (std::size_t i = 0; i < vectors->size(); ++i) {
    const json& item = (*vectors)[i];
    const std::string where = "vectors[" + std::to_string(i) + "]";
    if (!item.is_object() || item.size() != 2 || !item.contains("id") || !item.contains("vector")) {
      return bad_request<InsertRequest>(where + R"( must be {"id": ..., "vector": [...]})");
    }
    std::uint64_t id = 0;
    if (!get_uint(item["id"], std::numeric_limits<std::uint64_t>::max() - 1, id)) {
      return bad_request<InsertRequest>(where + ".id must be an integer in [0, 2^64 - 2]");
    }
    req.ids.push_back(id);
    if (!append_vector(item["vector"], dim, where + ".vector", req.rows, out)) {
      return out;
    }
  }
  out.ok = true;
  return out;
}

Decoded<InsertRequest> decode_bulk(std::string_view body, std::uint32_t dim, const Limits& limits) {
  constexpr std::size_t kHeader = 16;
  if (body.size() < kHeader || std::memcmp(body.data(), "VFB1", 4) != 0) {
    return bad_request<InsertRequest>("bulk body must start with the 16-byte VFB1 header");
  }
  std::uint64_t file_dim = 0;
  for (std::size_t i = 0; i < 4; ++i) {
    file_dim |= std::uint64_t{static_cast<unsigned char>(body[4 + i])} << (8U * i);
  }
  std::uint64_t count = 0;
  read_u64(body, 8, count);
  if (file_dim != dim) {
    return fail<InsertRequest>(
        400, "DIMENSION_MISMATCH",
        "bulk dim is " + std::to_string(file_dim) + ", expected " + std::to_string(dim));
  }
  if (count > limits.max_batch) {
    return limit_exceeded<InsertRequest>(std::to_string(count) +
                                         " vectors exceed the batch limit of " +
                                         std::to_string(limits.max_batch));
  }
  // count <= max_batch and dim <= 2^32 keep this product far from overflow.
  const std::uint64_t expected = kHeader + (count * 8U) + (count * dim * 4U);
  if (body.size() != expected) {
    return bad_request<InsertRequest>("bulk body is " + std::to_string(body.size()) +
                                      " bytes, expected " + std::to_string(expected));
  }
  Decoded<InsertRequest> out;
  const auto n = static_cast<std::size_t>(count);
  out.value.ids.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    read_u64(body, kHeader + (i * 8), out.value.ids[i]);
    if (out.value.ids[i] == kInvalidExternalId) {
      return bad_request<InsertRequest>("id " + std::to_string(i) + " is the reserved value");
    }
  }
  out.value.rows.resize(n * dim);
  const std::size_t offset = kHeader + (n * 8);
  for (std::size_t i = 0; i < out.value.rows.size(); ++i) {
    std::uint32_t bits = 0;
    for (std::size_t b = 0; b < 4; ++b) {
      bits |= std::uint32_t{static_cast<unsigned char>(body[offset + (i * 4) + b])} << (8U * b);
    }
    std::memcpy(&out.value.rows[i], &bits, sizeof(bits));
  }
  out.ok = true;
  return out;
}

Decoded<SearchRequest> decode_search(std::string_view body, std::uint32_t dim, bool batch,
                                     const Limits& limits) {
  Decoded<SearchRequest> out;
  json doc;
  if (!parse_object(body, {batch ? "vectors" : "vector", "k", "ef_search"}, doc, out)) {
    return out;
  }
  SearchRequest& req = out.value;
  if (!read_search_params(doc, limits, req.params, out)) {
    return out;
  }
  if (!batch) {
    const auto vector = doc.find("vector");
    if (vector == doc.end()) {
      return bad_request<SearchRequest>("'vector' is required");
    }
    req.count = 1;
    if (!append_vector(*vector, dim, "vector", req.vectors, out)) {
      return out;
    }
    out.ok = true;
    return out;
  }
  const auto vectors = doc.find("vectors");
  if (vectors == doc.end() || !vectors->is_array()) {
    return bad_request<SearchRequest>("'vectors' (array) is required");
  }
  if (vectors->size() > limits.max_batch) {
    return limit_exceeded<SearchRequest>(std::to_string(vectors->size()) +
                                         " queries exceed the batch limit of " +
                                         std::to_string(limits.max_batch));
  }
  req.count = vectors->size();
  req.vectors.reserve(req.count * dim);
  for (std::size_t i = 0; i < req.count; ++i) {
    if (!append_vector((*vectors)[i], dim, "vectors[" + std::to_string(i) + "]", req.vectors,
                       out)) {
      return out;
    }
  }
  out.ok = true;
  return out;
}

std::string encode_config(const CollectionConfig& c) {
  json index = {{"type", to_string(c.index)}};
  if (c.index == IndexType::Hnsw) {
    index["M"] = c.hnsw.M;
    index["ef_construction"] = c.hnsw.ef_construction;
    index["ef_search"] = c.hnsw.ef_search;
    index["max_level"] = c.hnsw.max_level;
    index["seed"] = c.hnsw.seed;
    index["concurrency"] = to_string(c.concurrency);
  }
  const json body = {{"dim", c.dim},
                     {"metric", to_string(c.metric)},
                     {"normalize", c.effective_normalize()},
                     {"index", index}};
  return body.dump();
}

std::string encode_collection(std::string_view name, const Collection& collection) {
  json body = json::parse(encode_config(collection.config()));
  body["name"] = name;
  body["count"] = collection.size();
  return body.dump();
}

std::string encode_stats(std::string_view name, const Collection& collection) {
  const CollectionStats s = collection.stats();
  const double deleted_ratio =
      s.row_count == 0 ? 0.0
                       : static_cast<double>(s.deleted_count) / static_cast<double>(s.row_count);
  const json body = {{"name", name},
                     {"count", s.live_count},
                     {"deleted", s.deleted_count},
                     {"rows", s.row_count},
                     {"deleted_ratio", deleted_ratio},
                     {"dim", s.dim},
                     {"metric", to_string(s.metric)},
                     {"index", to_string(s.index)},
                     {"normalized", s.normalized},
                     {"simd", to_string(s.simd)},
                     {"memory",
                      {{"vectors_bytes", s.memory.vectors_bytes},
                       {"mapped_vectors_bytes", s.memory.mapped_vectors_bytes},
                       {"labels_bytes", s.memory.labels_bytes},
                       {"id_map_bytes_estimate", s.memory.id_map_bytes_estimate},
                       {"tombstone_bytes", s.memory.tombstone_bytes},
                       {"index_bytes", s.memory.index_bytes},
                       {"total_bytes", s.memory.total_bytes()}}}};
  return body.dump();
}

std::string encode_neighbors(std::span<const Neighbor> neighbors) {
  json list = json::array();
  for (const Neighbor& n : neighbors) {
    list.push_back({{"id", n.id}, {"distance", n.distance}});
  }
  return list.dump();
}

std::string encode_vector(ExternalId id, std::span<const float> vector) {
  const json body = {{"id", id}, {"vector", std::vector<float>(vector.begin(), vector.end())}};
  return body.dump();
}

std::string format_ms(double milliseconds) {
  std::array<char, 32> buf{};
  const int n = std::snprintf(buf.data(), buf.size(), "%.3f", milliseconds);
  return {buf.data(), static_cast<std::size_t>(n > 0 ? n : 0)};
}

}  // namespace vf::server
