// Canonical .vfidx writer (docs/storage-format.md).

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <vector>

#include "collection/index_file.hpp"
#include "core/assert.hpp"
#include "index/hnsw/hnsw_backend.hpp"
#include "index/hnsw/hnsw_io.hpp"
#include "storage/format.hpp"

namespace vf::detail {

namespace {

struct PlannedSection {
  format::SectionType type;
  std::uint64_t size;
  std::uint64_t alignment;
  std::function<void(BinaryWriter&)> write;
};

format::Metadata make_metadata(const CollectionState& s, const HnswGraph* graph) {
  format::Metadata m;
  const CollectionConfig& c = s.config;
  m.dim = c.dim;
  m.metric = static_cast<std::uint8_t>(c.metric);
  m.index_type = static_cast<std::uint8_t>(c.index);
  m.normalize = c.normalize ? 1 : 0;
  m.node_count = s.vectors.size();
  m.live_count = s.ids.size();
  m.m = c.hnsw.M;
  m.m0 = c.hnsw.max_links_level0();
  m.ef_construction = c.hnsw.ef_construction;
  m.ef_search = c.hnsw.ef_search;
  m.max_level_cap = c.hnsw.max_level;
  m.seed = c.hnsw.seed;
  if (graph != nullptr && graph->entry().valid()) {
    m.entry_point = graph->entry().id;
    m.max_level = graph->entry().level;
  }
  m.creator = s.creator;
  m.created_unix_ms = s.created_unix_ms;
  return m;
}

}  // namespace

Status write_index(const CollectionState& s, ByteSink& sink) {
  const std::uint64_t n = s.vectors.size();
  const std::uint32_t dim = s.config.dim;
  VF_CHECK(s.ids.label_count() == n, "write_index: labels and rows out of sync");
  const HnswGraph* graph = nullptr;
  if (s.backend->type() == IndexType::Hnsw) {
    graph = &static_cast<const HnswBackend&>(*s.backend).graph();
  }
  const format::Metadata meta = make_metadata(s, graph);
  const bool has_tombstones = s.deleted.count() != 0;
  const std::uint64_t tombstone_words = (n + 63U) / 64U;

  std::vector<PlannedSection> plan;
  plan.push_back({format::SectionType::Metadata, format::metadata_size(meta),
                  format::kSectionAlignment,
                  [&meta](BinaryWriter& w) { format::encode_metadata(w, meta); }});
  plan.push_back({format::SectionType::Labels, n * 8, format::kSectionAlignment,
                  [&s](BinaryWriter& w) { w.array(s.ids.labels()); }});
  if (has_tombstones) {
    plan.push_back({format::SectionType::Tombstones, tombstone_words * 8, format::kSectionAlignment,
                    [&s, tombstone_words](BinaryWriter& w) {
                      w.array(s.deleted.words().first(static_cast<std::size_t>(tombstone_words)));
                    }});
  }
  if (graph != nullptr) {
    const std::uint64_t stride0 = std::uint64_t{graph->capacity(0)} + 1;
    plan.push_back({format::SectionType::Levels, n, format::kSectionAlignment,
                    [graph](BinaryWriter& w) { hnsw_io::write_levels(w, *graph); }});
    plan.push_back({format::SectionType::L0Links, n * stride0 * 4, format::kSectionAlignment,
                    [graph](BinaryWriter& w) { hnsw_io::write_l0_links(w, *graph); }});
    plan.push_back({format::SectionType::UpperIndex, n * 8, format::kSectionAlignment,
                    [graph](BinaryWriter& w) { hnsw_io::write_upper_index(w, *graph); }});
    plan.push_back({format::SectionType::UpperLinks, hnsw_io::upper_link_words(*graph) * 4,
                    format::kSectionAlignment,
                    [graph](BinaryWriter& w) { hnsw_io::write_upper_links(w, *graph); }});
  }
  plan.push_back({format::SectionType::Vectors, n * dim * 4, format::kVectorAlignment,
                  [&s, dim](BinaryWriter& w) {
                    for (std::size_t c = 0; c < s.vectors.chunk_count(); ++c) {
                      const std::size_t rows = s.vectors.rows_in_chunk(c);
                      w.array(std::span<const float>(s.vectors.chunk_data(c), rows * dim));
                    }
                  }});

  // Layout (sizes are bounded by in-memory data, so alignment cannot overflow).
  const auto aligned = [](std::uint64_t value, std::uint64_t alignment) {
    const std::optional<std::uint64_t> result = format::align_up(value, alignment);
    VF_CHECK(result.has_value(), "write_index: layout overflow");
    return result.value_or(0);
  };
  std::vector<format::SectionEntry> entries(plan.size());
  std::uint64_t position = format::kHeaderSize;
  for (std::size_t i = 0; i < plan.size(); ++i) {
    position = aligned(position, plan[i].alignment);
    entries[i].type = static_cast<std::uint32_t>(plan[i].type);
    entries[i].offset = position;
    entries[i].size = plan[i].size;
    position += plan[i].size;
  }
  const std::uint64_t table_offset = aligned(position, format::kSectionAlignment);
  const std::uint64_t file_size = table_offset + (entries.size() * format::kSectionEntrySize);

  BinaryWriter w(sink);
  w.zeros(format::kHeaderSize);
  for (std::size_t i = 0; i < plan.size(); ++i) {
    w.zeros(entries[i].offset - w.position());
    w.begin_crc();
    plan[i].write(w);
    if (!w.status().ok()) {
      return w.status();
    }
    VF_CHECK(w.position() == entries[i].offset + entries[i].size,
             "write_index: section size does not match its plan");
    entries[i].crc32c = w.crc();
  }
  w.zeros(table_offset - w.position());
  w.begin_crc();
  for (const format::SectionEntry& e : entries) {
    format::encode_section_entry(w, e);
  }
  const std::uint32_t table_crc = w.crc();
  if (!w.status().ok()) {
    return w.status();
  }
  VF_CHECK(w.position() == file_size, "write_index: file size does not match its plan");

  format::FileHeader header;
  header.flags = (graph != nullptr ? format::kFlagHasGraph : 0) |
                 (has_tombstones ? format::kFlagHasTombstones : 0) |
                 (s.normalized ? format::kFlagVectorsNormalized : 0);
  header.file_size = file_size;
  header.section_table_offset = table_offset;
  header.section_count = static_cast<std::uint32_t>(entries.size());
  header.section_table_crc32c = table_crc;
  const std::array<std::byte, format::kHeaderSize> encoded = format::encode_header(header);
  return sink.write_at(0, encoded);
}

}  // namespace vf::detail
