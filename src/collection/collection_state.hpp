#pragma once

// Everything a collection owns. Members are declared in dependency order: the backend holds
// references to `vectors` and `deleted`, so it is declared (and therefore destroyed) after them.
// A CollectionState is heap-allocated and never moved, keeping those references valid.

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <vectorforge/config.hpp>

#include "index/index_backend.hpp"
#include "simd/kernels.hpp"
#include "storage/id_map.hpp"
#include "storage/tombstones.hpp"
#include "storage/vector_store.hpp"

namespace vf::detail {

struct CollectionState {
  CollectionState(const CollectionConfig& cfg, VectorStore store, const KernelTable& table)
      : config(cfg),
        normalized(cfg.effective_normalize()),
        kernels(&table),
        vectors(std::move(store)) {}

  CollectionState(const CollectionState&) = delete;
  CollectionState& operator=(const CollectionState&) = delete;
  CollectionState(CollectionState&&) = delete;
  CollectionState& operator=(CollectionState&&) = delete;
  ~CollectionState() = default;

  CollectionConfig config;
  bool normalized;
  const KernelTable* kernels;
  // Provenance, persisted and preserved across save/load (so re-saving is byte-identical).
  std::string creator;
  std::uint64_t created_unix_ms = 0;
  VectorStore vectors;
  IdMap ids;
  TombstoneSet deleted;
  std::unique_ptr<IndexBackend> backend;
};

}  // namespace vf::detail
