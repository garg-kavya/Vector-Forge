#include "index/hnsw/hnsw_io.hpp"

#include <string>
#include <vector>

#include "core/checked_math.hpp"
#include "storage/format.hpp"

namespace vf::detail::hnsw_io {

namespace {

std::uint32_t get_u32(std::span<const std::byte> bytes, std::size_t word) noexcept {
  const std::size_t o = word * 4;
  return static_cast<std::uint32_t>(bytes[o]) | (static_cast<std::uint32_t>(bytes[o + 1]) << 8U) |
         (static_cast<std::uint32_t>(bytes[o + 2]) << 16U) |
         (static_cast<std::uint32_t>(bytes[o + 3]) << 24U);
}

std::uint64_t get_u64(std::span<const std::byte> bytes, std::size_t index) noexcept {
  const std::size_t o = index * 8;
  std::uint64_t v = 0;
  for (std::size_t i = 0; i < 8; ++i) {
    v |= static_cast<std::uint64_t>(bytes[o + i]) << (8U * i);
  }
  return v;
}

Status corrupt(std::string_view section, const std::string& what) {
  return Status::corrupt_data(std::string(section) + ": " + what);
}

// Writes one list as (1 + capacity) words: count, ids, then empty slots.
void write_list(BinaryWriter& out, LinkView links, std::uint32_t capacity,
                std::vector<std::uint32_t>& buffer) {
  buffer.assign(std::size_t{capacity} + 1, format::kEmptySlot);
  buffer[0] = links.size();
  for (std::uint32_t j = 0; j < links.size(); ++j) {
    buffer[1 + j] = links[j];
  }
  out.array(std::span<const std::uint32_t>(buffer));
}

}  // namespace

std::uint64_t upper_link_words(const HnswGraph& graph) noexcept {
  std::uint64_t words = 0;
  for (std::uint64_t i = 0; i < graph.node_count(); ++i) {
    words += std::uint64_t{graph.level(static_cast<InternalId>(i))} * (1U + graph.m());
  }
  return words;
}

void write_levels(BinaryWriter& out, const HnswGraph& graph) {
  std::vector<std::uint8_t> levels(static_cast<std::size_t>(graph.node_count()));
  for (std::size_t i = 0; i < levels.size(); ++i) {
    levels[i] = graph.level(static_cast<InternalId>(i));
  }
  out.array(std::span<const std::uint8_t>(levels));
}

void write_l0_links(BinaryWriter& out, const HnswGraph& graph) {
  std::vector<std::uint32_t> buffer;
  for (std::uint64_t i = 0; i < graph.node_count(); ++i) {
    write_list(out, graph.links(static_cast<InternalId>(i), 0), graph.capacity(0), buffer);
  }
}

void write_upper_index(BinaryWriter& out, const HnswGraph& graph) {
  std::uint64_t offset = 0;
  for (std::uint64_t i = 0; i < graph.node_count(); ++i) {
    const std::uint8_t level = graph.level(static_cast<InternalId>(i));
    if (level == 0) {
      out.u64(format::kNoUpperBlock);
    } else {
      out.u64(offset);
      offset += std::uint64_t{level} * (1U + graph.m());
    }
  }
}

void write_upper_links(BinaryWriter& out, const HnswGraph& graph) {
  std::vector<std::uint32_t> buffer;
  for (std::uint64_t i = 0; i < graph.node_count(); ++i) {
    const auto id = static_cast<InternalId>(i);
    for (std::uint8_t l = 1; l <= graph.level(id); ++l) {
      write_list(out, graph.links(id, l), graph.capacity(l), buffer);
    }
  }
}

std::optional<std::uint64_t> upper_link_words(std::span<const std::byte> levels,
                                              std::uint32_t m) noexcept {
  std::uint64_t words = 0;
  const std::uint64_t stride = std::uint64_t{m} + 1;
  for (const std::byte b : levels) {
    const std::optional<std::uint64_t> block =
        checked_mul(std::uint64_t{static_cast<std::uint8_t>(b)}, stride);
    const std::optional<std::uint64_t> sum = block ? checked_add(words, *block) : std::nullopt;
    if (!sum) {
      return std::nullopt;
    }
    words = *sum;
  }
  return words;
}

Result<HnswGraph> decode(const GraphSections& s, const GraphShape& shape) {
  const std::uint64_t n = shape.node_count;
  const std::uint64_t m = shape.m;
  const std::uint64_t m0 = 2 * m;
  const std::uint64_t stride0 = m0 + 1;
  const std::uint64_t stride_upper = m + 1;

  const std::optional<std::uint64_t> l0_bytes = checked_mul(n, stride0, std::uint64_t{4});
  const std::optional<std::uint64_t> index_bytes = checked_mul(n, std::uint64_t{8});
  if (s.levels.size() != n) {
    return corrupt("LEVELS", "size does not match node_count");
  }
  if (!l0_bytes || s.l0_links.size() != *l0_bytes) {
    return corrupt("L0_LINKS", "size does not match node_count and M");
  }
  if (!index_bytes || s.upper_index.size() != *index_bytes) {
    return corrupt("UPPER_INDEX", "size does not match node_count");
  }

  std::uint8_t top = 0;
  for (std::uint64_t i = 0; i < n; ++i) {
    const auto level = static_cast<std::uint8_t>(s.levels[static_cast<std::size_t>(i)]);
    if (level > shape.max_level_cap) {
      return corrupt("LEVELS", "node " + std::to_string(i) + " has level " + std::to_string(level) +
                                   " above max_level_cap " + std::to_string(shape.max_level_cap));
    }
    top = level > top ? level : top;
  }
  if (n == 0) {
    if (shape.entry_point != format::kEmptySlot || shape.entry_level != 0) {
      return corrupt("METADATA", "empty graph must have no entry point");
    }
  } else if (shape.entry_point >= n ||
             static_cast<std::uint8_t>(s.levels[shape.entry_point]) != shape.entry_level ||
             shape.entry_level != top) {
    return corrupt("METADATA", "entry point is not a node of the top level");
  }

  const std::optional<std::uint64_t> upper_words = upper_link_words(s.levels, shape.m);
  const std::optional<std::uint64_t> upper_bytes =
      upper_words ? checked_mul(*upper_words, std::uint64_t{4}) : std::nullopt;
  if (!upper_bytes || s.upper_links.size() != *upper_bytes) {
    return corrupt("UPPER_LINKS", "size does not match LEVELS and M");
  }

  Result<HnswGraph> created = HnswGraph::create({.m = shape.m, .max_level = shape.max_level_cap});
  if (!created.ok()) {
    return corrupt("METADATA", created.status().message());
  }
  HnswGraph graph = std::move(created).value();

  // stamp[id] == list number while id is being checked in that list: duplicate detection in
  // O(total links) without sorting.
  std::vector<std::uint64_t> stamp(static_cast<std::size_t>(n), 0);
  std::uint64_t list_number = 0;
  std::vector<InternalId> ids;
  ids.reserve(static_cast<std::size_t>(m0));
  std::uint64_t expected_offset = 0;

  for (std::uint64_t i = 0; i < n; ++i) {
    const auto node = static_cast<InternalId>(i);
    const auto level = static_cast<std::uint8_t>(s.levels[static_cast<std::size_t>(i)]);
    const std::uint64_t upper_offset = get_u64(s.upper_index, static_cast<std::size_t>(i));
    if (level == 0 ? upper_offset != format::kNoUpperBlock : upper_offset != expected_offset) {
      return corrupt("UPPER_INDEX", "node " + std::to_string(i) + " has a non-canonical offset");
    }
    VF_RETURN_IF_ERROR(graph.add_node(level));

    for (std::uint8_t l = 0; l <= level; ++l) {
      const bool base = l == 0;
      const std::span<const std::byte> section = base ? s.l0_links : s.upper_links;
      const std::string_view name = base ? "L0_LINKS" : "UPPER_LINKS";
      const std::uint64_t capacity = base ? m0 : m;
      const std::uint64_t first =
          base ? i * stride0 : expected_offset + ((std::uint64_t{l} - 1) * stride_upper);
      const std::uint32_t count = get_u32(section, static_cast<std::size_t>(first));
      const std::string where = "node " + std::to_string(i) + " level " + std::to_string(l);
      if (count > capacity) {
        return corrupt(name, where + " has " + std::to_string(count) + " links (capacity " +
                                 std::to_string(capacity) + ")");
      }
      ++list_number;
      ids.clear();
      for (std::uint64_t j = 0; j < capacity; ++j) {
        const std::uint32_t id = get_u32(section, static_cast<std::size_t>(first + 1 + j));
        if (j >= count) {
          if (id != format::kEmptySlot) {
            return corrupt(name, where + " has a non-empty unused slot");
          }
          continue;
        }
        if (id >= n) {
          return corrupt(name, where + " links to nonexistent node " + std::to_string(id));
        }
        if (id == node) {
          return corrupt(name, where + " links to itself");
        }
        if (static_cast<std::uint8_t>(s.levels[id]) < l) {
          return corrupt(name,
                         where + " links to node " + std::to_string(id) + " below that level");
        }
        if (stamp[id] == list_number) {
          return corrupt(name, where + " links to node " + std::to_string(id) + " twice");
        }
        stamp[id] = list_number;
        ids.push_back(id);
      }
      graph.set_links(node, l, ids);
    }
    if (level > 0) {
      expected_offset += std::uint64_t{level} * stride_upper;
    }
  }
  if (n > 0) {
    graph.set_entry({.id = shape.entry_point, .level = shape.entry_level});
  }
  return graph;
}

}  // namespace vf::detail::hnsw_io
