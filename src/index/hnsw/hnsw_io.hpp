#pragma once

// Encoding and validation of the HNSW graph sections of an index file (docs/storage-format.md):
//   LEVELS       node_count x u8
//   L0_LINKS     node_count x (1 + M0) x u32   count, ids, unused slots = 0xFFFFFFFF
//   UPPER_INDEX  node_count x u64              word offset of the node's block in UPPER_LINKS,
//                                              0xFFFFFFFFFFFFFFFF for level-0 nodes
//   UPPER_LINKS  concatenated blocks, level x (1 + M) x u32 per node, in node order

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

#include <vectorforge/status.hpp>

#include "index/hnsw/hnsw_graph.hpp"
#include "storage/binary_io.hpp"

namespace vf::detail::hnsw_io {

// Words of UPPER_LINKS for `graph`.
[[nodiscard]] std::uint64_t upper_link_words(const HnswGraph& graph) noexcept;

void write_levels(BinaryWriter& out, const HnswGraph& graph);
void write_l0_links(BinaryWriter& out, const HnswGraph& graph);
void write_upper_index(BinaryWriter& out, const HnswGraph& graph);
void write_upper_links(BinaryWriter& out, const HnswGraph& graph);

// Words of UPPER_LINKS implied by LEVELS (nullopt on overflow). Precondition: levels validated.
[[nodiscard]] std::optional<std::uint64_t> upper_link_words(std::span<const std::byte> levels,
                                                            std::uint32_t m) noexcept;

struct GraphSections {
  std::span<const std::byte> levels;
  std::span<const std::byte> l0_links;
  std::span<const std::byte> upper_index;
  std::span<const std::byte> upper_links;
};

struct GraphShape {
  std::uint64_t node_count = 0;
  std::uint32_t m = 0;
  std::uint8_t max_level_cap = 0;
  std::uint32_t entry_point = 0;
  std::uint8_t entry_level = 0;
};

// Validates the graph sections completely (sizes, level range, entry point, every list: count <=
// capacity, ids in range, no self-links, no duplicates, linked nodes have a sufficient level,
// unused slots 0xFFFFFFFF, canonical UPPER_INDEX offsets) and builds the graph. Never reads out of
// bounds. Errors: CorruptData naming the section and node. Throws std::bad_alloc.
[[nodiscard]] Result<HnswGraph> decode(const GraphSections& sections, const GraphShape& shape);

}  // namespace vf::detail::hnsw_io
