#pragma once

// Serialisation of a CollectionState to and from the .vfidx format (docs/storage-format.md).

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <vectorforge/config.hpp>
#include <vectorforge/status.hpp>

#include "collection/collection_state.hpp"
#include "storage/binary_io.hpp"
#include "storage/format.hpp"

namespace vf::detail {

// Writes the canonical encoding of `state`: sections METADATA, LABELS, [TOMBSTONES], [LEVELS,
// L0_LINKS, UPPER_INDEX, UPPER_LINKS], VECTORS (page-aligned), then the section table, with zero
// padding. Equal states produce identical bytes. The sink must support write_at (header patch).
// Errors: whatever the sink reports.
[[nodiscard]] Status write_index(const CollectionState& state, ByteSink& sink);

struct ReadOptions {
  Verify verify = Verify::Full;  // Auto is not accepted here; callers resolve it
  // Non-null: `file` stays valid while `owner` lives, and vectors are served from it in place.
  // Null: vectors are copied to heap chunks.
  std::shared_ptr<const void> owner;
  Concurrency concurrency = Concurrency::Concurrent;  // runtime setting of the loaded collection
};

// Validates `file` completely (docs/storage-format.md, "Validation") and builds a state.
// Errors: CorruptData (any inconsistency, naming section and field), UnsupportedVersion (other
// format major, big-endian file, unknown required section). Never reads out of bounds.
// Throws std::bad_alloc; allocations are bounded by a small multiple of the file size.
[[nodiscard]] Result<std::unique_ptr<CollectionState>> read_index(std::span<const std::byte> file,
                                                                  const ReadOptions& options);

// Header, section table and metadata of a file that passes header and table validation
// (for `vectorforge info`). Does not validate section contents beyond METADATA.
struct IndexSummary {
  format::FileHeader header;
  std::vector<format::SectionEntry> sections;
  format::Metadata metadata;
};
[[nodiscard]] Result<IndexSummary> summarize_index(std::span<const std::byte> file);

}  // namespace vf::detail
