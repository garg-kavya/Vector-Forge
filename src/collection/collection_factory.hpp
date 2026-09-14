#pragma once

// Internal entry points that construct or inspect Collections outside the public API
// (in-memory loading for fuzzing and tests).

#include <cstddef>
#include <memory>
#include <span>

#include <vectorforge/collection.hpp>
#include <vectorforge/config.hpp>

#include "collection/collection_state.hpp"
#include "storage/binary_io.hpp"

namespace vf::detail {

struct CollectionFactory {
  // Validates and loads an index from memory, copying all data (Verify::Auto means Full).
  [[nodiscard]] static Result<std::unique_ptr<Collection>> load_from_memory(
      std::span<const std::byte> file, Verify verify);
  // Serialises into memory.
  [[nodiscard]] static Status save_to(const Collection& collection, ByteSink& sink);
  [[nodiscard]] static const CollectionState& state(const Collection& collection) noexcept;
};

}  // namespace vf::detail
