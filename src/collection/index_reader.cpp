// Validating .vfidx reader (docs/storage-format.md, "Validation"). The file is untrusted input:
// every offset and size is checked with overflow-safe arithmetic before use, and every failure is
// reported as a Status, never an assertion or an out-of-bounds read.

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "collection/index_file.hpp"
#include "core/checked_math.hpp"
#include "core/validation.hpp"
#include "index/flat_backend.hpp"
#include "index/hnsw/hnsw_backend.hpp"
#include "index/hnsw/hnsw_io.hpp"
#include "simd/dispatch.hpp"
#include "simd/kernels.hpp"
#include "storage/crc32c.hpp"

namespace vf::detail {

namespace {

using format::SectionEntry;
using format::SectionType;

Status corrupt(const std::string& what) {
  return Status::corrupt_data(what);
}

struct ParsedFile {
  format::FileHeader header;
  std::vector<SectionEntry> entries;
  std::array<std::optional<std::size_t>, format::kMaxKnownSectionType + 1> by_type{};
};

std::span<const std::byte> section_bytes(std::span<const std::byte> file, const SectionEntry& e) {
  // Bounds were validated in parse_header_and_table.
  return file.subspan(static_cast<std::size_t>(e.offset), static_cast<std::size_t>(e.size));
}

// Steps 1 and 2: header, section table, section bounds, overlaps and zero padding.
Result<ParsedFile> parse_header_and_table(std::span<const std::byte> file) {
  if (file.size() < format::kHeaderSize) {
    return corrupt("file is smaller than the 64-byte header");
  }
  const std::span<const std::byte, format::kHeaderSize> head = file.first<format::kHeaderSize>();
  if (!std::equal(format::kMagic.begin(), format::kMagic.end(), head.begin())) {
    return corrupt("not a VectorForge index (bad magic)");
  }
  ByteReader probe(head);
  std::uint32_t endian = 0;
  static_cast<void>(probe.seek(12));
  static_cast<void>(probe.read_u32(endian));
  if (endian == format::kEndianTagSwapped) {
    return Status::unsupported_version("index file has big-endian byte order");
  }
  if (endian != format::kEndianTag) {
    return corrupt("header: invalid endian tag");
  }
  ParsedFile parsed;
  parsed.header = format::decode_header(head);
  const format::FileHeader& h = parsed.header;
  if (h.format_major != format::kFormatMajor) {
    return Status::unsupported_version(
        "index format major version " + std::to_string(h.format_major) +
        " (supported: " + std::to_string(format::kFormatMajor) + ")");
  }
  if (crc32c(file.first(format::kHeaderCrcCoverage)) != h.header_crc32c) {
    return corrupt("header: checksum mismatch");
  }
  if (!format::header_reserved_zero(head)) {
    return corrupt("header: reserved fields are not zero");
  }
  if (h.file_size != file.size()) {
    return corrupt("header: file_size " + std::to_string(h.file_size) + " but the file has " +
                   std::to_string(file.size()) + " bytes (truncated or extended)");
  }
  if ((h.flags & ~format::kKnownFlags) != 0) {
    return corrupt("header: unknown flags");
  }
  if (h.section_count == 0 || h.section_count > format::kMaxSections) {
    return corrupt("header: section_count " + std::to_string(h.section_count) + " out of range");
  }
  const std::uint64_t table_size = std::uint64_t{h.section_count} * format::kSectionEntrySize;
  const std::optional<std::uint64_t> table_end = checked_add(h.section_table_offset, table_size);
  if (h.section_table_offset < format::kHeaderSize || !table_end || *table_end > file.size() ||
      h.section_table_offset % format::kSectionAlignment != 0) {
    return corrupt("header: section table out of bounds");
  }
  const std::span<const std::byte> table = file.subspan(
      static_cast<std::size_t>(h.section_table_offset), static_cast<std::size_t>(table_size));
  if (crc32c(table) != h.section_table_crc32c) {
    return corrupt("section table: checksum mismatch");
  }

  ByteReader in(table);
  parsed.entries.resize(h.section_count);
  for (SectionEntry& e : parsed.entries) {
    bool reserved_zero = false;
    if (!format::decode_section_entry(in, e, reserved_zero) || !reserved_zero) {
      return corrupt("section table: malformed entry");
    }
    const std::optional<std::uint64_t> end = checked_add(e.offset, e.size);
    if (e.offset < format::kHeaderSize || !end || *end > file.size() ||
        e.offset % format::kSectionAlignment != 0) {
      return corrupt("section table: section type " + std::to_string(e.type) + " out of bounds");
    }
    if ((e.flags & ~format::kSectionOptional) != 0) {
      return corrupt("section table: unknown section flags");
    }
    if (e.type == 0 || e.type > format::kMaxKnownSectionType) {
      if ((e.flags & format::kSectionOptional) == 0) {
        return Status::unsupported_version("index contains unknown required section type " +
                                           std::to_string(e.type));
      }
      continue;  // unknown optional section: skipped
    }
    std::optional<std::size_t>& slot = parsed.by_type[e.type];
    if (slot) {
      return corrupt("section table: duplicate " +
                     std::string(format::to_string(static_cast<SectionType>(e.type))));
    }
    slot = static_cast<std::size_t>(&e - parsed.entries.data());
  }

  // Regions (header, sections, table) must not overlap, and every byte outside them must be zero.
  std::vector<std::pair<std::uint64_t, std::uint64_t>> regions;
  regions.reserve(parsed.entries.size() + 2);
  regions.emplace_back(0, format::kHeaderSize);
  for (const SectionEntry& e : parsed.entries) {
    regions.emplace_back(e.offset, e.offset + e.size);
  }
  regions.emplace_back(h.section_table_offset, *table_end);
  std::sort(regions.begin(), regions.end());
  std::uint64_t covered = 0;
  for (const auto& [begin, end] : regions) {
    if (begin < covered) {
      return corrupt("section table: sections overlap");
    }
    const std::span<const std::byte> gap =
        file.subspan(static_cast<std::size_t>(covered), static_cast<std::size_t>(begin - covered));
    if (std::any_of(gap.begin(), gap.end(), [](std::byte b) { return b != std::byte{0}; })) {
      return corrupt("padding between sections is not zero");
    }
    covered = end;
  }
  const std::span<const std::byte> tail = file.subspan(static_cast<std::size_t>(covered));
  if (std::any_of(tail.begin(), tail.end(), [](std::byte b) { return b != std::byte{0}; })) {
    return corrupt("padding after the last region is not zero");
  }
  return parsed;
}

const SectionEntry* find(const ParsedFile& p, SectionType type) noexcept {
  const std::optional<std::size_t>& slot = p.by_type[static_cast<std::size_t>(type)];
  return slot ? &p.entries[*slot] : nullptr;
}

Status verify_crc(std::span<const std::byte> file, const SectionEntry& e) {
  if (crc32c(section_bytes(file, e)) != e.crc32c) {
    return corrupt(std::string(format::to_string(static_cast<SectionType>(e.type))) +
                   ": checksum mismatch");
  }
  return {};
}

// Step 3: metadata values.
Status validate_metadata(const format::Metadata& m, const format::FileHeader& h) {
  if (m.dim < 1 || m.dim > kMaxDim) {
    return corrupt("METADATA: dim " + std::to_string(m.dim) + " out of range");
  }
  if (!is_valid(static_cast<Metric>(m.metric))) {
    return corrupt("METADATA: invalid metric " + std::to_string(m.metric));
  }
  if (!is_valid(static_cast<IndexType>(m.index_type))) {
    return corrupt("METADATA: invalid index type " + std::to_string(m.index_type));
  }
  if (m.normalize > 1) {
    return corrupt("METADATA: invalid normalize flag");
  }
  if (m.node_count > kMaxVectorsPerCollection) {
    return corrupt("METADATA: node_count " + std::to_string(m.node_count) + " out of range");
  }
  if (m.live_count > m.node_count) {
    return corrupt("METADATA: live_count exceeds node_count");
  }
  const bool normalized = m.normalize == 1 || static_cast<Metric>(m.metric) == Metric::Cosine;
  if (normalized != ((h.flags & format::kFlagVectorsNormalized) != 0)) {
    return corrupt("header: VECTORS_NORMALIZED flag contradicts METADATA");
  }
  const bool hnsw = static_cast<IndexType>(m.index_type) == IndexType::Hnsw;
  if (hnsw != ((h.flags & format::kFlagHasGraph) != 0)) {
    return corrupt("header: HAS_GRAPH flag contradicts the index type");
  }
  if (m.max_level_cap > HnswParams::kMaxLevelCap || m.max_level > m.max_level_cap) {
    return corrupt("METADATA: max_level out of range");
  }
  if (hnsw) {
    HnswParams p;
    p.M = m.m;
    p.ef_construction = m.ef_construction;
    p.ef_search = m.ef_search;
    p.max_level = m.max_level_cap;
    p.seed = m.seed;
    if (const Status st = p.validate(); !st.ok()) {
      return corrupt("METADATA: " + st.message());
    }
    if (m.m0 != 2 * m.m) {
      return corrupt("METADATA: M0 must equal 2 * M");
    }
  } else if (m.entry_point != format::kEmptySlot || m.max_level != 0) {
    return corrupt("METADATA: Flat index with a graph entry point");
  }
  return {};
}

Status expect_size(const SectionEntry* e, SectionType type, std::optional<std::uint64_t> expected) {
  if (!expected || e->size != *expected) {
    return corrupt(std::string(format::to_string(type)) + ": size " + std::to_string(e->size) +
                   " does not match METADATA");
  }
  return {};
}

std::vector<std::uint64_t> read_u64_array(std::span<const std::byte> bytes) {
  std::vector<std::uint64_t> out(bytes.size() / 8);
  if (!out.empty()) {
    std::memcpy(out.data(), bytes.data(), out.size() * 8);  // little-endian host
  }
  return out;
}

}  // namespace

Result<std::unique_ptr<CollectionState>> read_index(std::span<const std::byte> file,
                                                    const ReadOptions& options) {
  VF_CHECK(options.verify != Verify::Auto, "read_index: Verify::Auto must be resolved by caller");
  Result<ParsedFile> parsed_or = parse_header_and_table(file);
  if (!parsed_or.ok()) {
    return parsed_or.status();
  }
  const ParsedFile& p = parsed_or.value();
  const format::FileHeader& h = p.header;

  // Presence of sections.
  const SectionEntry* meta_e = find(p, SectionType::Metadata);
  const SectionEntry* vectors_e = find(p, SectionType::Vectors);
  const SectionEntry* labels_e = find(p, SectionType::Labels);
  if (meta_e == nullptr || vectors_e == nullptr || labels_e == nullptr) {
    return corrupt("section table: METADATA, VECTORS and LABELS are required");
  }
  const SectionEntry* tomb_e = find(p, SectionType::Tombstones);
  if ((tomb_e != nullptr) != ((h.flags & format::kFlagHasTombstones) != 0)) {
    return corrupt("header: HAS_TOMBSTONES flag contradicts the section table");
  }
  const bool has_graph = (h.flags & format::kFlagHasGraph) != 0;
  const std::array<SectionType, 4> graph_types = {SectionType::Levels, SectionType::L0Links,
                                                  SectionType::UpperIndex, SectionType::UpperLinks};
  for (const SectionType t : graph_types) {
    if ((find(p, t) != nullptr) != has_graph) {
      return corrupt("section table: " + std::string(format::to_string(t)) +
                     (has_graph ? " is missing" : " present without HAS_GRAPH"));
    }
  }
  if (vectors_e->offset % format::kVectorAlignment != 0) {
    return corrupt("VECTORS: offset is not page-aligned");
  }

  // Checksums of everything but the vectors (Metadata and Full).
  if (options.verify != Verify::None) {
    for (const SectionEntry& e : p.entries) {
      if (e.type != static_cast<std::uint32_t>(SectionType::Vectors)) {
        VF_RETURN_IF_ERROR(verify_crc(file, e));
      }
    }
  }

  // Step 3: metadata.
  Result<format::Metadata> meta_or = format::decode_metadata(section_bytes(file, *meta_e));
  if (!meta_or.ok()) {
    return meta_or.status();
  }
  const format::Metadata& m = meta_or.value();
  VF_RETURN_IF_ERROR(validate_metadata(m, h));
  const std::uint64_t n = m.node_count;

  // Step 4: section sizes.
  VF_RETURN_IF_ERROR(expect_size(vectors_e, SectionType::Vectors,
                                 checked_mul(n, std::uint64_t{m.dim}, std::uint64_t{4})));
  VF_RETURN_IF_ERROR(expect_size(labels_e, SectionType::Labels, checked_mul(n, std::uint64_t{8})));
  if (tomb_e != nullptr) {
    VF_RETURN_IF_ERROR(
        expect_size(tomb_e, SectionType::Tombstones, ((n + 63U) / 64U) * std::uint64_t{8}));
  }

  // Step 5: vector checksum.
  const std::span<const std::byte> vector_bytes = section_bytes(file, *vectors_e);
  if (options.verify == Verify::Full) {
    VF_RETURN_IF_ERROR(verify_crc(file, *vectors_e));
  }

  // Tombstones and labels.
  TombstoneSet deleted;
  if (tomb_e != nullptr) {
    std::optional<TombstoneSet> restored =
        TombstoneSet::restore(read_u64_array(section_bytes(file, *tomb_e)), n);
    if (!restored) {
      return corrupt("TOMBSTONES: bits set beyond the last row");
    }
    if (restored->count() == 0) {
      return corrupt("TOMBSTONES: section present but empty");
    }
    deleted = std::move(*restored);
  }
  if (n - deleted.count() != m.live_count) {
    return corrupt("METADATA: live_count does not match TOMBSTONES");
  }
  deleted.ensure_size(n);
  Result<IdMap> ids = IdMap::restore(read_u64_array(section_bytes(file, *labels_e)), deleted);
  if (!ids.ok()) {
    return ids.status();
  }

  // Step 6: graph.
  std::optional<HnswGraph> graph;
  if (has_graph) {
    const SectionEntry* levels_e = find(p, SectionType::Levels);
    const SectionEntry* l0_e = find(p, SectionType::L0Links);
    const SectionEntry* index_e = find(p, SectionType::UpperIndex);
    const SectionEntry* upper_e = find(p, SectionType::UpperLinks);
    // Presence was checked against HAS_GRAPH above.
    VF_CHECK(levels_e != nullptr && l0_e != nullptr && index_e != nullptr && upper_e != nullptr,
             "read_index: graph sections missing after validation");
    const hnsw_io::GraphSections sections{.levels = section_bytes(file, *levels_e),
                                          .l0_links = section_bytes(file, *l0_e),
                                          .upper_index = section_bytes(file, *index_e),
                                          .upper_links = section_bytes(file, *upper_e)};
    const hnsw_io::GraphShape shape{.node_count = n,
                                    .m = m.m,
                                    .max_level_cap = m.max_level_cap,
                                    .entry_point = m.entry_point,
                                    .entry_level = m.max_level};
    Result<HnswGraph> decoded = hnsw_io::decode(sections, shape);
    if (!decoded.ok()) {
      return decoded.status();
    }
    graph.emplace(std::move(decoded).value());
  }

  // Vectors: in place (mapped) when possible, otherwise copied. Copied vectors are always checked
  // for non-finite or out-of-range components; mapped ones only with Verify::Full, which touches
  // every page anyway.
  const std::uint32_t dim = m.dim;
  const bool in_place = options.owner != nullptr &&
                        reinterpret_cast<std::uintptr_t>(vector_bytes.data()) % alignof(float) == 0;
  const bool check_values = !in_place || options.verify == Verify::Full;
  Result<VectorStore> store_or = Status::internal("unset");
  if (in_place) {
    // The file bytes are an implicit-lifetime float array created by the mapping.
    const auto* rows = reinterpret_cast<const float*>(vector_bytes.data());
    if (check_values && find_invalid_component(std::span<const float>(
                            rows, vector_bytes.size() / 4)) != vector_bytes.size() / 4) {
      return corrupt("VECTORS: non-finite or out-of-range component");
    }
    store_or = VectorStore::create_mapped({.dim = dim}, options.owner, rows, n);
  } else {
    store_or = VectorStore::create({.dim = dim});
    if (store_or.ok()) {
      VectorStore& store = store_or.value();
      VF_RETURN_IF_ERROR(store.reserve(n));
      std::vector<float> row(dim);
      for (std::uint64_t r = 0; r < n; ++r) {
        std::memcpy(row.data(), vector_bytes.data() + static_cast<std::size_t>(r * dim * 4),
                    std::size_t{dim} * 4);
        if (find_invalid_component(row) != row.size()) {
          return corrupt("VECTORS: row " + std::to_string(r) +
                         " has a non-finite or out-of-range component");
        }
        VF_RETURN_IF_ERROR(store.append(row).status());
      }
    }
  }
  if (!store_or.ok()) {
    return store_or.status();
  }

  CollectionConfig config;
  config.dim = dim;
  config.metric = static_cast<Metric>(m.metric);
  config.normalize = m.normalize == 1;
  config.index = static_cast<IndexType>(m.index_type);
  config.concurrency = options.concurrency;
  config.hnsw.M = m.m;
  config.hnsw.ef_construction = m.ef_construction;
  config.hnsw.ef_search = m.ef_search;
  config.hnsw.max_level = m.max_level_cap;
  config.hnsw.seed = m.seed;

  if (const Status& simd = kernel_selection().status; !simd.ok()) {
    return simd;
  }
  const KernelTable& kernels = detail::kernels();
  auto state = std::make_unique<CollectionState>(config, std::move(store_or).value(), kernels);
  state->ids = std::move(ids).value();
  state->deleted = std::move(deleted);
  state->creator = m.creator;
  state->created_unix_ms = m.created_unix_ms;
  if (graph) {
    Result<std::unique_ptr<HnswBackend>> backend =
        HnswBackend::create_loaded(state->vectors, state->deleted, config.metric, state->normalized,
                                   kernels, config.hnsw, std::move(*graph));
    if (!backend.ok()) {
      return corrupt("METADATA: " + backend.status().message());
    }
    state->backend = std::move(backend).value();
  } else {
    state->backend = std::make_unique<FlatBackend>(state->vectors, state->deleted, config.metric,
                                                   state->normalized, kernels);
  }
  return state;
}

Result<IndexSummary> summarize_index(std::span<const std::byte> file) {
  Result<ParsedFile> parsed = parse_header_and_table(file);
  if (!parsed.ok()) {
    return parsed.status();
  }
  const SectionEntry* meta_e = find(parsed.value(), SectionType::Metadata);
  if (meta_e == nullptr) {
    return corrupt("section table: METADATA is missing");
  }
  Result<format::Metadata> meta = format::decode_metadata(section_bytes(file, *meta_e));
  if (!meta.ok()) {
    return meta.status();
  }
  return IndexSummary{.header = parsed.value().header,
                      .sections = std::move(parsed.value().entries),
                      .metadata = std::move(meta).value()};
}

}  // namespace vf::detail
