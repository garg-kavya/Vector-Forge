// Fuzz target for the index reader (docs/DESIGN.md §12.3, §15.1). The input is an arbitrary byte
// string treated as an index file. Loading must never crash or trip a sanitizer. A file that loads
// must support searching and inserting, and its re-encoding must load with full verification.
//
// Built as a libFuzzer binary with -DVF_BUILD_FUZZERS=ON (Clang), and as a corpus replay test
// everywhere (replay_main.cpp).

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <vector>

#include <vectorforge/collection.hpp>

#include "collection/collection_factory.hpp"
#include "storage/binary_io.hpp"

namespace {

void check(bool condition) {
  if (!condition) {
    std::abort();  // reported by the fuzzer as a crash
  }
}

void exercise(vf::Collection& c) {
  const std::uint32_t dim = c.config().dim;
  std::vector<float> v(dim, 0.25F);
  vf::SearchParams p;
  p.k = 7;
  p.ef_search = 12;
  for (int round = 0; round < 2; ++round) {
    const auto hits = c.search(v, p);
    check(hits.ok());
    check(hits.value().size() <= c.size());
    v[0] += 1.0F;
  }
  check(c.add(0xF00DULL, v, {.upsert = true}).ok());
  static_cast<void>(c.remove(0xF00DULL));

  vf::detail::MemorySink sink;
  check(vf::detail::CollectionFactory::save_to(c, sink).ok());
  const auto reloaded =
      vf::detail::CollectionFactory::load_from_memory(sink.bytes(), vf::Verify::Full);
  check(reloaded.ok());
  check(reloaded.value()->size() == c.size());
}

}  // namespace

// NOLINTNEXTLINE(readability-identifier-naming): name required by libFuzzer
extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  const std::span<const std::byte> bytes(reinterpret_cast<const std::byte*>(data), size);
  for (const vf::Verify verify : {vf::Verify::None, vf::Verify::Full}) {
    auto loaded = vf::detail::CollectionFactory::load_from_memory(bytes, verify);
    if (loaded.ok()) {
      exercise(*loaded.value());
    }
  }
  return 0;
}
