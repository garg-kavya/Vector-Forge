#include "index/hnsw/hnsw_graph.hpp"

#include <algorithm>
#include <bit>
#include <string>
#include <utility>

namespace vf::detail {

namespace {

constexpr std::size_t kTargetChunkBytes = std::size_t{16} << 20U;  // 16 MiB
constexpr std::size_t kMaxNodesPerChunk = std::size_t{1} << 16U;

// Reserves room for one more element with geometric growth so that a following push_back cannot
// throw.
template <class T>
void reserve_one_more(std::vector<T>& v) {
  if (v.size() == v.capacity()) {
    v.reserve(std::max<std::size_t>(16, v.capacity() * 2));
  }
}

void put_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
  for (unsigned shift = 0; shift < 32; shift += 8) {
    out.push_back(static_cast<std::uint8_t>(value >> shift));
  }
}

void put_u64(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (unsigned shift = 0; shift < 64; shift += 8) {
    out.push_back(static_cast<std::uint8_t>(value >> shift));
  }
}

}  // namespace

Result<HnswGraph> HnswGraph::create(const Options& options) {
  if (options.m < 2) {
    return Status::invalid_argument("hnsw graph: m must be >= 2");
  }
  if (options.max_level < 1) {
    return Status::invalid_argument("hnsw graph: max_level must be >= 1");
  }
  const std::size_t stride0 = 1 + (2 * static_cast<std::size_t>(options.m));
  std::size_t nodes_per_chunk = options.nodes_per_chunk;
  if (nodes_per_chunk == 0) {
    const std::size_t fit = kTargetChunkBytes / (stride0 * sizeof(std::uint32_t));
    nodes_per_chunk = std::clamp<std::size_t>(std::bit_floor(std::max<std::size_t>(fit, 1)), 1,
                                              kMaxNodesPerChunk);
  } else if (!std::has_single_bit(nodes_per_chunk) || nodes_per_chunk > (std::size_t{1} << 31U)) {
    return Status::invalid_argument("hnsw graph: nodes_per_chunk must be a power of two <= 2^31");
  }
  std::size_t arena_words = options.arena_chunk_words;
  if (arena_words == 0) {
    arena_words = kDefaultArenaChunkWords;
  }
  const std::size_t max_block = static_cast<std::size_t>(options.max_level) * (1 + options.m);
  if (!std::has_single_bit(arena_words) || arena_words < max_block ||
      arena_words > (std::size_t{1} << 31U)) {
    return Status::invalid_argument(
        "hnsw graph: arena_chunk_words must be a power of two in [max_level * (1 + m), 2^31], "
        "got " +
        std::to_string(arena_words));
  }
  return HnswGraph(options, nodes_per_chunk, arena_words);
}

HnswGraph::HnswGraph(const Options& options, std::size_t nodes_per_chunk,
                     std::size_t arena_chunk_words)
    : m_(options.m),
      m0_(2 * options.m),
      max_level_(options.max_level),
      stride0_(1 + (2 * static_cast<std::size_t>(options.m))),
      stride_upper_(1 + static_cast<std::size_t>(options.m)),
      nodes_per_chunk_(nodes_per_chunk),
      chunk_shift_(static_cast<std::uint32_t>(std::countr_zero(nodes_per_chunk))),
      chunk_mask_(static_cast<std::uint32_t>(nodes_per_chunk - 1)),
      arena_chunk_words_(arena_chunk_words),
      arena_shift_(static_cast<std::uint32_t>(std::countr_zero(arena_chunk_words))),
      arena_mask_(static_cast<std::uint32_t>(arena_chunk_words - 1)),
      arena_used_(arena_chunk_words) {  // "full" so the first upper block opens a chunk
}

Status HnswGraph::add_node(std::uint8_t level) {
  if (level > max_level_) {
    return Status::invalid_argument("hnsw graph: level " + std::to_string(level) +
                                    " exceeds max_level " + std::to_string(max_level_));
  }
  const std::uint64_t id = levels_.size();
  if (id >= kMaxVectorsPerCollection) {
    return Status::resource_exhausted("hnsw graph: node id space exhausted");
  }

  // Phase 1: everything that can fail, without changing observable state.
  reserve_one_more(levels_);
  reserve_one_more(upper_);
  AtomicArray<std::uint32_t> new_l0;
  if ((id & chunk_mask_) == 0) {
    reserve_one_more(l0_chunks_);
    new_l0 = make_atomic_array<std::uint32_t>(nodes_per_chunk_ * stride0_);
  }
  const std::size_t block_words = static_cast<std::size_t>(level) * stride_upper_;
  AtomicArray<std::uint32_t> new_arena;
  bool opens_arena_chunk = false;
  if (level > 0 && arena_used_ + block_words > arena_chunk_words_) {
    const std::size_t max_chunks = std::size_t{1} << (32U - arena_shift_);
    if (arena_chunks_.size() >= max_chunks) {
      return Status::resource_exhausted("hnsw graph: upper-level arena exhausted");
    }
    reserve_one_more(arena_chunks_);
    new_arena = make_atomic_array<std::uint32_t>(arena_chunk_words_);
    opens_arena_chunk = true;
  }

  // Phase 2: commit (no allocation, cannot throw).
  if (new_l0) {
    l0_chunks_.push_back(std::move(new_l0));
  }
  std::uint32_t offset = 0;
  if (level > 0) {
    if (opens_arena_chunk) {
      arena_chunks_.push_back(std::move(new_arena));
      arena_used_ = 0;
    }
    offset = static_cast<std::uint32_t>(((arena_chunks_.size() - 1) << arena_shift_) + arena_used_);
    arena_used_ += block_words;
  }
  levels_.push_back(level);
  upper_.push_back(offset);
  return {};
}

HnswGraph::HnswGraph(HnswGraph&& other) noexcept
    : m_(other.m_),
      m0_(other.m0_),
      max_level_(other.max_level_),
      stride0_(other.stride0_),
      stride_upper_(other.stride_upper_),
      nodes_per_chunk_(other.nodes_per_chunk_),
      chunk_shift_(other.chunk_shift_),
      chunk_mask_(other.chunk_mask_),
      arena_chunk_words_(other.arena_chunk_words_),
      arena_shift_(other.arena_shift_),
      arena_mask_(other.arena_mask_),
      arena_used_(other.arena_used_),
      levels_(std::move(other.levels_)),
      upper_(std::move(other.upper_)),
      l0_chunks_(std::move(other.l0_chunks_)),
      arena_chunks_(std::move(other.arena_chunks_)),
      entry_(other.entry_.load()) {
  other.entry_.store(pack(Entry{}));
}

HnswGraph& HnswGraph::operator=(HnswGraph&& other) noexcept {
  if (this != &other) {
    m_ = other.m_;
    m0_ = other.m0_;
    max_level_ = other.max_level_;
    stride0_ = other.stride0_;
    stride_upper_ = other.stride_upper_;
    nodes_per_chunk_ = other.nodes_per_chunk_;
    chunk_shift_ = other.chunk_shift_;
    chunk_mask_ = other.chunk_mask_;
    arena_chunk_words_ = other.arena_chunk_words_;
    arena_shift_ = other.arena_shift_;
    arena_mask_ = other.arena_mask_;
    arena_used_ = other.arena_used_;
    levels_ = std::move(other.levels_);
    upper_ = std::move(other.upper_);
    l0_chunks_ = std::move(other.l0_chunks_);
    arena_chunks_ = std::move(other.arena_chunks_);
    entry_.store(other.entry_.load());
    other.entry_.store(pack(Entry{}));
  }
  return *this;
}

void HnswGraph::set_links(InternalId id, std::uint8_t level,
                          std::span<const InternalId> ids) noexcept {
  VF_ASSERT(ids.size() <= capacity(level), "HnswGraph::set_links: too many links");
  LinkWord* list = list_ptr(id, level);
  for (std::size_t i = 0; i < ids.size(); ++i) {
    list[1 + i].store(ids[i], std::memory_order_relaxed);
  }
  list[0].store(static_cast<std::uint32_t>(ids.size()), std::memory_order_relaxed);
}

bool HnswGraph::try_append_link(InternalId id, std::uint8_t level, InternalId neighbor) noexcept {
  LinkWord* list = list_ptr(id, level);
  const std::uint32_t count = list[0].load(std::memory_order_relaxed);
  if (count >= capacity(level)) {
    return false;
  }
  list[1 + count].store(neighbor, std::memory_order_relaxed);
  list[0].store(count + 1, std::memory_order_relaxed);
  return true;
}

std::size_t HnswGraph::bytes() const noexcept {
  return (l0_chunks_.size() * nodes_per_chunk_ * stride0_ * sizeof(std::uint32_t)) +
         (arena_chunks_.size() * arena_chunk_words_ * sizeof(std::uint32_t)) + levels_.capacity() +
         (upper_.capacity() * sizeof(std::uint32_t));
}

std::vector<std::uint8_t> HnswGraph::canonical_bytes() const {
  std::vector<std::uint8_t> out;
  put_u32(out, m_);
  put_u32(out, max_level_);
  put_u64(out, node_count());
  const Entry top = entry();
  put_u32(out, top.id);
  put_u32(out, top.level);
  for (std::size_t i = 0; i < levels_.size(); ++i) {
    const auto id = static_cast<InternalId>(i);
    out.push_back(levels_[i]);
    for (std::uint8_t level = 0; level <= levels_[i]; ++level) {
      const LinkView view = links(id, level);
      put_u32(out, view.size());
      for (std::uint32_t j = 0; j < view.size(); ++j) {
        put_u32(out, view[j]);
      }
    }
  }
  return out;
}

}  // namespace vf::detail
