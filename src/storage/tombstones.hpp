#pragma once

// Deletion markers for rows (docs/DESIGN.md §9.9). Rows are never physically removed while a
// collection state is alive; searches skip tombstoned rows and compaction rebuilds without them.
//
// Thread safety: thread-compatible (Phase 2). Phase 6b switches words to std::atomic.

#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <vectorforge/types.hpp>

#include "core/assert.hpp"

namespace vf::detail {

class TombstoneSet {
 public:
  // Ensures rows [0, rows) can be marked. Never shrinks. Throws std::bad_alloc.
  void ensure_size(std::uint64_t rows) {
    const auto words = static_cast<std::size_t>((rows + 63U) / 64U);
    if (words > words_.size()) {
      words_.resize(words, 0);
    }
  }

  // Marks `id`; returns true if it was not already marked. Precondition: id within ensure_size().
  bool set(InternalId id) noexcept {
    VF_ASSERT(id / 64U < words_.size(), "TombstoneSet::set out of range");
    std::uint64_t& word = words_[id / 64U];
    const std::uint64_t mask = std::uint64_t{1} << (id % 64U);
    if ((word & mask) != 0U) {
      return false;
    }
    word |= mask;
    ++count_;
    return true;
  }

  [[nodiscard]] bool test(InternalId id) const noexcept {
    const std::size_t index = id / 64U;
    return index < words_.size() && ((words_[index] >> (id % 64U)) & 1U) != 0U;
  }

  // Bitset words (bit i of word w marks row w * 64 + i); may extend past the last row.
  [[nodiscard]] std::span<const std::uint64_t> words() const noexcept { return words_; }

  // Tombstones for `rows` rows from persisted words (exactly ceil(rows / 64) of them). Returns
  // nullopt if the word count is wrong or a bit beyond the last row is set.
  [[nodiscard]] static std::optional<TombstoneSet> restore(std::vector<std::uint64_t> words,
                                                           std::uint64_t rows) {
    if (words.size() != (rows + 63U) / 64U) {
      return std::nullopt;
    }
    const auto tail_bits = static_cast<unsigned>(rows % 64U);
    if (tail_bits != 0 && (words.back() >> tail_bits) != 0U) {
      return std::nullopt;
    }
    TombstoneSet set;
    for (const std::uint64_t w : words) {
      set.count_ += static_cast<std::uint64_t>(std::popcount(w));
    }
    set.words_ = std::move(words);
    return set;
  }

  [[nodiscard]] std::uint64_t count() const noexcept { return count_; }
  [[nodiscard]] bool any() const noexcept { return count_ != 0; }
  [[nodiscard]] std::size_t bytes() const noexcept {
    return words_.capacity() * sizeof(std::uint64_t);
  }

 private:
  std::vector<std::uint64_t> words_;
  std::uint64_t count_ = 0;
};

}  // namespace vf::detail
