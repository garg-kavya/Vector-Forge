#pragma once

// Fundamental vocabulary types shared by every VectorForge component.

#include <cstdint>
#include <limits>
#include <string_view>

#include <vectorforge/status.hpp>

namespace vf {

// Distance metric of a collection. Internally every metric is a "lower is better" distance:
//   L2           -> squared Euclidean distance  sum((a_i - b_i)^2)
//   InnerProduct -> -dot(a, b)
//   Cosine       -> 1 - cos(a, b)  (vectors are normalised on write and on query)
enum class Metric : std::uint8_t {
  L2 = 0,
  InnerProduct = 1,
  Cosine = 2,
};

enum class IndexType : std::uint8_t {
  Flat = 0,
  Hnsw = 1,
};

// User-facing vector identifier.
using ExternalId = std::uint64_t;

// Reserved external id: never accepted on insert; marks padding slots in batch search output.
inline constexpr ExternalId kInvalidExternalId = std::numeric_limits<ExternalId>::max();

// Dense per-collection identifier (row number in vector storage and node id in the graph).
using InternalId = std::uint32_t;

inline constexpr InternalId kInvalidInternalId = std::numeric_limits<InternalId>::max();

// Largest supported dimensionality.
inline constexpr std::uint32_t kMaxDim = 65536;

// Largest number of vectors per collection: every InternalId except kInvalidInternalId.
inline constexpr std::uint64_t kMaxVectorsPerCollection = std::uint64_t{kInvalidInternalId};

// Largest accepted absolute value of a vector component. With |x| <= 1e16 and dim <= 65536,
// sums of squares and dot products stay below FLT_MAX, so no kernel can overflow to +/-inf and
// produce NaN (inf - inf) from finite input. NaN distances would break heap ordering.
inline constexpr float kMaxAbsComponent = 1e16F;

// One search hit. Results are ordered by ascending distance; equal distances are ordered by
// insertion order (older vectors first).
struct Neighbor {
  ExternalId id = 0;
  float distance = 0.0F;

  friend bool operator==(const Neighbor&, const Neighbor&) = default;
};

[[nodiscard]] bool is_valid(Metric metric) noexcept;
[[nodiscard]] bool is_valid(IndexType type) noexcept;

// "l2", "ip", "cosine".
[[nodiscard]] std::string_view to_string(Metric metric) noexcept;
// "flat", "hnsw".
[[nodiscard]] std::string_view to_string(IndexType type) noexcept;

// Accepts "l2", "ip", "inner_product", "cosine" (case-sensitive).
[[nodiscard]] Result<Metric> parse_metric(std::string_view text);
// Accepts "flat", "hnsw" (case-sensitive).
[[nodiscard]] Result<IndexType> parse_index_type(std::string_view text);

}  // namespace vf
