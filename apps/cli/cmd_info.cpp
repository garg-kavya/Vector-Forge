#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include <vectorforge/collection.hpp>

#include "collection/collection_factory.hpp"
#include "collection/index_file.hpp"
#include "commands.hpp"
#include "index/hnsw/hnsw_backend.hpp"
#include "index/hnsw/hnsw_validator.hpp"
#include "storage/format.hpp"
#include "storage/mapped_file.hpp"
#include "util/timer.hpp"

namespace vf::cli {

namespace {

const detail::HnswGraph* graph_of(const Collection& col) {
  const detail::CollectionState& s = detail::CollectionFactory::state(col);
  if (s.backend->type() != IndexType::Hnsw) {
    return nullptr;
  }
  return &static_cast<const detail::HnswBackend&>(*s.backend).graph();
}

std::string section_name(std::uint32_t type) {
  if (type >= 1 && type <= detail::format::kMaxKnownSectionType) {
    return std::string(detail::format::to_string(static_cast<detail::format::SectionType>(type)));
  }
  return "unknown(" + std::to_string(type) + ")";
}

}  // namespace

Status run_info(const std::filesystem::path& index, std::ostream& log) {
  Result<detail::MappedFile> mapped = detail::MappedFile::open(index);
  if (!mapped.ok()) {
    return mapped.status();
  }
  Result<detail::IndexSummary> summary = detail::summarize_index(mapped.value().data());
  if (!summary.ok()) {
    return {summary.status().code(), index.string() + ": " + summary.status().message()};
  }
  const detail::IndexSummary& s = summary.value();
  const detail::format::Metadata& m = s.metadata;
  log << index.string() << "\n"
      << "  format " << s.header.format_major << "." << s.header.format_minor << ", "
      << s.header.file_size << " bytes, flags 0x" << std::hex << s.header.flags << std::dec
      << "\n  created by '" << m.creator << "' at unix ms " << m.created_unix_ms << "\n"
      << "  dim " << m.dim << ", metric " << to_string(static_cast<Metric>(m.metric)) << ", index "
      << to_string(static_cast<IndexType>(m.index_type)) << ", normalize " << int{m.normalize}
      << "\n  rows " << m.node_count << ", live " << m.live_count << ", deleted "
      << (m.node_count - std::min(m.live_count, m.node_count)) << "\n";
  if (static_cast<IndexType>(m.index_type) == IndexType::Hnsw) {
    log << "  hnsw: M " << m.m << ", M0 " << m.m0 << ", ef_construction " << m.ef_construction
        << ", ef_search " << m.ef_search << ", max_level_cap " << int{m.max_level_cap}
        << ", top level " << int{m.max_level} << ", entry point " << m.entry_point << ", seed "
        << m.seed << "\n";
  }
  log << "  sections:\n";
  for (const detail::format::SectionEntry& e : s.sections) {
    log << "    " << std::left << std::setw(12) << section_name(e.type) << std::right << " offset "
        << std::setw(12) << e.offset << "  size " << std::setw(12) << e.size << "  crc32c 0x"
        << std::hex << std::setw(8) << std::setfill('0') << e.crc32c << std::setfill(' ')
        << std::dec << "\n";
  }

  const detail::Stopwatch open_timer;
  Result<std::unique_ptr<Collection>> loaded = Collection::load(index);
  if (!loaded.ok()) {
    log << "  load failed: " << loaded.status().to_string() << "\n";
    return loaded.status();
  }
  log << "  loads (mmap, metadata checksums) in " << open_timer.elapsed_seconds() << " s\n";
  if (const detail::HnswGraph* graph = graph_of(*loaded.value())) {
    const std::vector<std::uint64_t> histogram = detail::HnswValidator(*graph).level_histogram();
    log << "  level histogram:";
    for (std::size_t l = 0; l < histogram.size(); ++l) {
      if (histogram[l] != 0) {
        log << " L" << l << "=" << histogram[l];
      }
    }
    log << "\n";
  }
  return {};
}

Status run_verify(const std::filesystem::path& index, std::ostream& log) {
  const detail::Stopwatch timer;
  Result<std::unique_ptr<Collection>> loaded =
      Collection::load(index, {.use_mmap = false, .verify = Verify::Full});
  if (!loaded.ok()) {
    log << "FAILED: " << loaded.status().to_string() << "\n";
    return loaded.status();
  }
  if (const detail::HnswGraph* graph = graph_of(*loaded.value())) {
    const detail::HnswValidator validator(*graph);
    const Status invariants = validator.check_invariants();
    if (!invariants.ok()) {
      log << "FAILED: " << invariants.to_string() << "\n";
      return Status::corrupt_data(invariants.message());
    }
    const detail::HnswReachability reach = validator.reachability();
    for (std::size_t l = 0; l < reach.unreachable.size(); ++l) {
      if (reach.unreachable[l] != 0) {
        log << "note: " << reach.unreachable[l] << " nodes unreachable on level " << l << "\n";
      }
    }
  }
  log << "OK: " << index.string() << " (checksums, structure and graph invariants verified in "
      << timer.elapsed_seconds() << " s)\n";
  return {};
}

}  // namespace vf::cli
