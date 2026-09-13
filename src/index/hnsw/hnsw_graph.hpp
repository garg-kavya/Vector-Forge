#pragma once

// HNSW graph storage (docs/DESIGN.md §9.1, §9.3).
//
// Level 0: every node owns a fixed-stride list { count; ids[M0] } (M0 = 2M) in chunked arrays, so
// node i's list lives at chunk[i >> shift] + (i & mask) * stride with no indirection.
// Levels >= 1: nodes with level l >= 1 own a block of l lists { count; ids[M] } in a chunked,
// append-only arena; upper_[i] is the arena offset of that block.
//
// All addresses are stable once allocated. Link lists are read through LinkView, so the Phase 6b
// switch to atomic slots is local to this header.
//
// Thread safety (Phase 3): thread-compatible. Const access may be concurrent; mutation requires
// exclusive access.

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <vectorforge/status.hpp>
#include <vectorforge/types.hpp>

#include "core/aligned_alloc.hpp"
#include "core/assert.hpp"

namespace vf::detail {

// Read-only view of one link list: size() neighbour ids.
class LinkView {
 public:
  explicit constexpr LinkView(const std::uint32_t* list) noexcept : list_(list) {}

  [[nodiscard]] constexpr std::uint32_t size() const noexcept { return list_[0]; }
  [[nodiscard]] constexpr bool empty() const noexcept { return list_[0] == 0; }
  [[nodiscard]] constexpr InternalId operator[](std::uint32_t i) const noexcept {
    return list_[1 + i];
  }
  [[nodiscard]] std::span<const InternalId> ids() const noexcept { return {list_ + 1, list_[0]}; }

 private:
  const std::uint32_t* list_;
};

class HnswGraph {
 public:
  struct Entry {
    InternalId id = kInvalidInternalId;
    std::uint8_t level = 0;

    [[nodiscard]] constexpr bool valid() const noexcept { return id != kInvalidInternalId; }
    friend bool operator==(const Entry&, const Entry&) = default;
  };

  struct Options {
    std::uint32_t m = 16;
    std::uint8_t max_level = 16;
    // Level-0 lists per chunk; 0 selects a default of about 16 MiB per chunk. Power of two.
    std::size_t nodes_per_chunk = 0;
    // Words per upper-level arena chunk; 0 selects kDefaultArenaChunkWords. Must be at least
    // max_level * (1 + m).
    std::size_t arena_chunk_words = 0;
  };

  static constexpr std::size_t kDefaultArenaChunkWords = std::size_t{1} << 16U;

  // Errors: InvalidArgument (m < 2, max_level < 1, bad chunk sizes).
  [[nodiscard]] static Result<HnswGraph> create(const Options& options);

  HnswGraph(const HnswGraph&) = delete;
  HnswGraph& operator=(const HnswGraph&) = delete;
  HnswGraph(HnswGraph&&) noexcept = default;
  HnswGraph& operator=(HnswGraph&&) noexcept = default;
  ~HnswGraph() = default;

  // Appends node node_count() with top level `level` and empty link lists.
  // Errors: InvalidArgument (level > max_level), ResourceExhausted (id space or arena exhausted).
  // Throws std::bad_alloc; on any failure the graph is unchanged.
  [[nodiscard]] Status add_node(std::uint8_t level);

  [[nodiscard]] std::uint64_t node_count() const noexcept { return levels_.size(); }
  [[nodiscard]] std::uint32_t m() const noexcept { return m_; }
  [[nodiscard]] std::uint8_t max_level() const noexcept { return max_level_; }
  // Link capacity of a list on `level`: 2M on level 0, M above.
  [[nodiscard]] std::uint32_t capacity(std::uint8_t level) const noexcept {
    return level == 0 ? m0_ : m_;
  }

  // Precondition: id < node_count().
  [[nodiscard]] std::uint8_t level(InternalId id) const noexcept {
    VF_ASSERT(id < levels_.size(), "HnswGraph::level: id out of range");
    return levels_[id];
  }

  // Precondition: id < node_count(), level <= this->level(id).
  [[nodiscard]] LinkView links(InternalId id, std::uint8_t level) const noexcept {
    return LinkView(list_ptr(id, level));
  }

  // Overwrites a list: ids first, then the count (§9.6: stale ids past the count are left in
  // place). Preconditions: ids.size() <= capacity(level); every id valid, distinct and != `id`.
  void set_links(InternalId id, std::uint8_t level, std::span<const InternalId> ids) noexcept;

  // Appends `neighbor` if the list has room; returns false when full.
  bool try_append_link(InternalId id, std::uint8_t level, InternalId neighbor) noexcept;

  [[nodiscard]] Entry entry() const noexcept { return entry_; }
  void set_entry(Entry entry) noexcept { entry_ = entry; }

  // Bytes allocated for graph structures.
  [[nodiscard]] std::size_t bytes() const noexcept;

  // Canonical little-endian encoding of the logical graph (parameters, entry point, levels and the
  // first `count` ids of every list). Two graphs are structurally identical iff their encodings
  // are equal; used by determinism tests until the Phase 4 file format exists.
  [[nodiscard]] std::vector<std::uint8_t> canonical_bytes() const;

 private:
  HnswGraph(const Options& options, std::size_t nodes_per_chunk, std::size_t arena_chunk_words);

  [[nodiscard]] const std::uint32_t* list_ptr(InternalId id, std::uint8_t level) const noexcept {
    VF_ASSERT(id < levels_.size() && level <= levels_[id], "HnswGraph: list out of range");
    if (level == 0) {
      return l0_chunks_[id >> chunk_shift_].get() + ((id & chunk_mask_) * stride0_);
    }
    const std::uint32_t offset = upper_[id];
    return arena_chunks_[offset >> arena_shift_].get() + (offset & arena_mask_) +
           ((static_cast<std::size_t>(level) - 1) * stride_upper_);
  }
  [[nodiscard]] std::uint32_t* list_ptr_mut(InternalId id, std::uint8_t level) noexcept;

  std::uint32_t m_;
  std::uint32_t m0_;
  std::uint8_t max_level_;
  std::size_t stride0_;       // 1 + m0 words
  std::size_t stride_upper_;  // 1 + m words
  std::size_t nodes_per_chunk_;
  std::uint32_t chunk_shift_;
  std::uint32_t chunk_mask_;
  std::size_t arena_chunk_words_;
  std::uint32_t arena_shift_;
  std::uint32_t arena_mask_;
  std::size_t arena_used_ = 0;  // words used in the last arena chunk

  std::vector<std::uint8_t> levels_;
  std::vector<std::uint32_t> upper_;  // arena offset per node (meaningful iff level >= 1)
  std::vector<AlignedArray<std::uint32_t>> l0_chunks_;
  std::vector<AlignedArray<std::uint32_t>> arena_chunks_;
  Entry entry_{};
};

}  // namespace vf::detail
