#include <algorithm>
#include <atomic>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <span>
#include <string>
#include <system_error>
#include <utility>

#include <vectorforge/catalog.hpp>

#include "collection/snapshot.hpp"
#include "storage/atomic_file.hpp"
#include "storage/flat_json.hpp"

namespace vf {

namespace detail {
namespace {

constexpr std::string_view kCollectionsDir = "collections";
constexpr std::string_view kConfigName = "config.json";
constexpr std::string_view kDroppedMarker = "DROPPED";
constexpr std::size_t kMaxConfigBytes = 4096;

std::string encode_config(const CollectionConfig& c) {
  std::string out = "{\n";
  out += R"(  "format": 1,)"
         "\n";
  out += "  \"dim\": " + std::to_string(c.dim) + ",\n";
  out += R"(  "metric": ")" + std::string(to_string(c.metric)) + "\",\n";
  out += std::string("  \"normalize\": ") + (c.normalize ? "true" : "false") + ",\n";
  out += R"(  "index": ")" + std::string(to_string(c.index)) + "\",\n";
  out += "  \"M\": " + std::to_string(c.hnsw.M) + ",\n";
  out += "  \"ef_construction\": " + std::to_string(c.hnsw.ef_construction) + ",\n";
  out += "  \"ef_search\": " + std::to_string(c.hnsw.ef_search) + ",\n";
  out += "  \"max_level\": " + std::to_string(c.hnsw.max_level) + ",\n";
  out += "  \"seed\": " + std::to_string(c.hnsw.seed) + "\n";
  out += "}\n";
  return out;
}

// Strict parser for encode_config() output (any key order; every key exactly once).
Result<CollectionConfig> decode_config(std::string_view text) {
  auto bad = [](const std::string& what) { return Status::corrupt_data("config.json: " + what); };
  FlatJsonScanner s(text);
  if (!s.consume('{')) {
    return bad("expected an object");
  }
  CollectionConfig c;
  std::set<std::string> seen;
  for (bool first = true;; first = false) {
    if (s.consume('}')) {
      break;
    }
    if (!first && !s.consume(',')) {
      return bad("expected ','");
    }
    std::string key;
    if (!s.string(key) || !s.consume(':') || !seen.insert(key).second) {
      return bad("expected a new key");
    }
    std::uint64_t number = 0;
    std::string text_value;
    bool ok = false;
    if (key == "format") {
      ok = s.number(number) && number == 1;
    } else if (key == "dim") {
      ok = s.number(number) && number <= kMaxDim;
      c.dim = static_cast<std::uint32_t>(number);
    } else if (key == "metric") {
      ok = s.string(text_value);
      if (ok) {
        Result<Metric> m = parse_metric(text_value);
        ok = m.ok();
        c.metric = ok ? m.value() : Metric::L2;
      }
    } else if (key == "normalize") {
      ok = s.boolean(c.normalize);
    } else if (key == "index") {
      ok = s.string(text_value);
      if (ok) {
        Result<IndexType> t = parse_index_type(text_value);
        ok = t.ok();
        c.index = ok ? t.value() : IndexType::Flat;
      }
    } else if (key == "M") {
      ok = s.number(number) && number <= HnswParams::kMaxM;
      c.hnsw.M = static_cast<std::uint32_t>(number);
    } else if (key == "ef_construction") {
      ok = s.number(number) && number <= HnswParams::kMaxEf;
      c.hnsw.ef_construction = static_cast<std::uint32_t>(number);
    } else if (key == "ef_search") {
      ok = s.number(number) && number <= HnswParams::kMaxEf;
      c.hnsw.ef_search = static_cast<std::uint32_t>(number);
    } else if (key == "max_level") {
      ok = s.number(number) && number <= HnswParams::kMaxLevelCap;
      c.hnsw.max_level = static_cast<std::uint8_t>(number);
    } else if (key == "seed") {
      ok = s.number(c.hnsw.seed);
    }
    if (!ok) {
      return bad("unknown or malformed key '" + key + "'");
    }
  }
  if (!s.at_end()) {
    return bad("trailing characters");
  }
  if (seen.size() != 10) {
    return bad("missing keys");
  }
  if (Status st = c.validate(); !st.ok()) {
    return bad(st.message());
  }
  return c;
}

Result<std::string> read_small_file(const std::filesystem::path& path, std::size_t limit) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return Status::io_error("cannot open " + path.string());
  }
  std::string text(limit + 1, '\0');
  in.read(text.data(), static_cast<std::streamsize>(text.size()));
  if (in.bad()) {
    return Status::io_error("cannot read " + path.string());
  }
  text.resize(static_cast<std::size_t>(in.gcount()));
  if (text.size() > limit) {
    return Status::corrupt_data(path.string() + ": larger than " + std::to_string(limit) +
                                " bytes");
  }
  return text;
}

Status write_text(const std::filesystem::path& path, const std::string& text) {
  return write_atomic(path, [&text](ByteSink& sink) {
    return sink.write(std::as_bytes(std::span<const char>(text.data(), text.size())));
  });
}

// State shared with the deleters of handed-out collections, which may outlive the Catalog.
struct CatalogShared {
  std::filesystem::path collections;
  std::mutex mutex;
  std::set<std::string> dropping;  // dropped names whose files still exist
};

struct CatalogSlot {
  std::shared_ptr<Collection> collection;
  std::shared_ptr<std::atomic<bool>> dropped;
  std::shared_ptr<std::mutex> snapshot_mutex;
  std::uint64_t generation = 0;
};

}  // namespace
}  // namespace detail

struct Catalog::Impl {
  std::filesystem::path data_dir;
  std::shared_ptr<detail::CatalogShared> shared;
  mutable std::shared_mutex mutex;
  std::map<std::string, detail::CatalogSlot, std::less<>> slots;

  // Wraps `owned` so that its files are deleted after the last reference if it was dropped.
  [[nodiscard]] detail::CatalogSlot make_slot(const std::string& name,
                                              std::unique_ptr<Collection> owned,
                                              std::uint64_t generation) const {
    detail::CatalogSlot slot;
    slot.dropped = std::make_shared<std::atomic<bool>>(false);
    slot.snapshot_mutex = std::make_shared<std::mutex>();
    slot.generation = generation;
    // shared_ptr's constructor calls the deleter if it throws, so `raw` cannot leak.
    Collection* raw = owned.release();
    slot.collection = std::shared_ptr<Collection>(
        raw, [shared = shared, dropped = slot.dropped, name](Collection* c) {
          delete c;  // NOLINT(cppcoreguidelines-owning-memory): released from a unique_ptr
          if (!dropped->load()) {
            return;
          }
          std::error_code ec;
          std::filesystem::remove_all(shared->collections / name, ec);
          const std::lock_guard<std::mutex> lock(shared->mutex);
          shared->dropping.erase(name);
        });
    return slot;
  }
};

Catalog::Catalog(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {
}
Catalog::~Catalog() = default;

bool Catalog::is_valid_name(std::string_view name) noexcept {
  return !name.empty() && name.size() <= 64 && std::all_of(name.begin(), name.end(), [](char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' ||
           c == '-';
  });
}

Result<std::unique_ptr<Catalog>> Catalog::open(const std::filesystem::path& data_dir,
                                               const LoadOptions& options) {
  auto impl = std::make_unique<Impl>();
  impl->data_dir = data_dir;
  impl->shared = std::make_shared<detail::CatalogShared>();
  impl->shared->collections = data_dir / detail::kCollectionsDir;
  std::error_code ec;
  std::filesystem::create_directories(impl->shared->collections, ec);
  if (ec) {
    return Status::io_error("cannot create " + impl->shared->collections.string() + ": " +
                            ec.message());
  }
  std::filesystem::directory_iterator it(impl->shared->collections, ec);
  if (ec) {
    return Status::io_error("cannot list " + impl->shared->collections.string() + ": " +
                            ec.message());
  }
  for (const std::filesystem::directory_entry& entry : it) {
    const std::string name = entry.path().filename().string();
    if (!entry.is_directory(ec)) {
      continue;
    }
    // Interrupted creations (".<name>.creating") and dropped collections are removed.
    if (!is_valid_name(name) ||
        std::filesystem::exists(entry.path() / detail::kDroppedMarker, ec)) {
      if (name.starts_with('.') || is_valid_name(name)) {
        std::filesystem::remove_all(entry.path(), ec);
      }
      continue;
    }
    auto fail = [&name](const Status& st) {
      return Status(st.code(), "collection '" + name + "': " + st.message());
    };
    std::unique_ptr<Collection> collection;
    std::uint64_t generation = 0;
    Result<detail::Manifest> manifest = detail::read_manifest(entry.path());
    if (manifest.ok()) {
      Result<std::unique_ptr<Collection>> loaded = detail::load_snapshot(entry.path(), options);
      if (!loaded.ok()) {
        return fail(loaded.status());
      }
      collection = std::move(loaded).value();
      generation = manifest.value().generation;
      static_cast<void>(detail::collect_garbage(entry.path()));
    } else if (manifest.status().code() == ErrorCode::NotFound) {
      Result<std::string> text =
          detail::read_small_file(entry.path() / detail::kConfigName, detail::kMaxConfigBytes);
      if (!text.ok()) {
        return fail(text.status());
      }
      Result<CollectionConfig> config = detail::decode_config(text.value());
      if (!config.ok()) {
        return fail(config.status());
      }
      config.value().concurrency = options.concurrency;
      Result<std::unique_ptr<Collection>> created = Collection::create(config.value());
      if (!created.ok()) {
        return fail(created.status());
      }
      collection = std::move(created).value();
    } else {
      return fail(manifest.status());
    }
    detail::CatalogSlot slot = impl->make_slot(name, std::move(collection), generation);
    impl->slots.emplace(name, std::move(slot));
  }
  return std::unique_ptr<Catalog>(new Catalog(std::move(impl)));
}

Result<std::shared_ptr<Collection>> Catalog::create(std::string_view name,
                                                    const CollectionConfig& config) {
  if (!is_valid_name(name)) {
    return Status::invalid_argument("collection name must match [A-Za-z0-9_-]{1,64}");
  }
  VF_RETURN_IF_ERROR(config.validate());
  const std::unique_lock<std::shared_mutex> lock(impl_->mutex);
  const std::string key(name);
  if (impl_->slots.contains(key)) {
    return Status::already_exists("collection '" + key + "' already exists");
  }
  {
    const std::lock_guard<std::mutex> shared_lock(impl_->shared->mutex);
    if (impl_->shared->dropping.contains(key)) {
      return Status::unavailable("collection '" + key +
                                 "' was dropped and is still in use; retry later");
    }
  }
  Result<std::unique_ptr<Collection>> created = Collection::create(config);
  if (!created.ok()) {
    return created.status();
  }
  // Written under a temporary name and renamed, so a crash never leaves a half-created collection.
  const std::filesystem::path final_dir = impl_->shared->collections / key;
  const std::filesystem::path temp_dir = impl_->shared->collections / ("." + key + ".creating");
  std::error_code ec;
  std::filesystem::remove_all(temp_dir, ec);
  std::filesystem::remove_all(final_dir, ec);  // leftovers of an earlier crash
  std::filesystem::create_directories(temp_dir, ec);
  if (ec) {
    return Status::io_error("cannot create " + temp_dir.string() + ": " + ec.message());
  }
  VF_RETURN_IF_ERROR(
      detail::write_text(temp_dir / detail::kConfigName, detail::encode_config(config)));
  VF_RETURN_IF_ERROR(detail::rename_replace(temp_dir, final_dir));
  VF_RETURN_IF_ERROR(detail::sync_directory(impl_->shared->collections));
  detail::CatalogSlot slot = impl_->make_slot(key, std::move(created).value(), 0);
  std::shared_ptr<Collection> handle = slot.collection;
  impl_->slots.emplace(key, std::move(slot));
  return handle;
}

Result<std::shared_ptr<Collection>> Catalog::get(std::string_view name) const {
  const std::shared_lock<std::shared_mutex> lock(impl_->mutex);
  const auto it = impl_->slots.find(name);
  if (it == impl_->slots.end()) {
    return Status::not_found("collection '" + std::string(name) + "' not found");
  }
  return it->second.collection;
}

Status Catalog::drop(std::string_view name) {
  detail::CatalogSlot slot;
  {
    const std::unique_lock<std::shared_mutex> lock(impl_->mutex);
    const auto it = impl_->slots.find(name);
    if (it == impl_->slots.end()) {
      return Status::not_found("collection '" + std::string(name) + "' not found");
    }
    const std::filesystem::path dir = impl_->shared->collections / it->first;
    // The marker makes the drop durable: open() deletes marked directories.
    VF_RETURN_IF_ERROR(detail::write_text(dir / detail::kDroppedMarker, "dropped\n"));
    {
      const std::lock_guard<std::mutex> shared_lock(impl_->shared->mutex);
      impl_->shared->dropping.insert(it->first);
    }
    it->second.dropped->store(true);
    slot = std::move(it->second);
    impl_->slots.erase(it);
  }
  // Waits for a snapshot in progress, then releases the catalog's reference outside the lock: the
  // files go when the last in-flight user releases the collection.
  const std::lock_guard<std::mutex> snapshot_lock(*slot.snapshot_mutex);
  return {};
}

std::vector<std::string> Catalog::list() const {
  const std::shared_lock<std::shared_mutex> lock(impl_->mutex);
  std::vector<std::string> names;
  names.reserve(impl_->slots.size());
  for (const auto& [name, slot] : impl_->slots) {
    names.push_back(name);
  }
  return names;
}

std::vector<CatalogEntry> Catalog::entries() const {
  std::vector<std::pair<std::string, detail::CatalogSlot>> copies;
  {
    const std::shared_lock<std::shared_mutex> lock(impl_->mutex);
    copies.assign(impl_->slots.begin(), impl_->slots.end());
  }
  std::vector<CatalogEntry> out;
  out.reserve(copies.size());
  for (const auto& [name, slot] : copies) {
    out.push_back({.name = name,
                   .config = slot.collection->config(),
                   .size = slot.collection->size(),
                   .last_generation = slot.generation});
  }
  return out;
}

std::size_t Catalog::size() const {
  const std::shared_lock<std::shared_mutex> lock(impl_->mutex);
  return impl_->slots.size();
}

Result<SnapshotInfo> Catalog::snapshot(std::string_view name) {
  std::shared_ptr<Collection> collection;
  std::shared_ptr<std::mutex> snapshot_mutex;
  {
    const std::shared_lock<std::shared_mutex> lock(impl_->mutex);
    const auto it = impl_->slots.find(name);
    if (it == impl_->slots.end()) {
      return Status::not_found("collection '" + std::string(name) + "' not found");
    }
    collection = it->second.collection;
    snapshot_mutex = it->second.snapshot_mutex;
  }
  const std::lock_guard<std::mutex> snapshot_lock(*snapshot_mutex);
  const std::filesystem::path dir = impl_->shared->collections / std::string(name);
  Result<std::uint64_t> generation = detail::save_snapshot(*collection, dir);
  if (!generation.ok()) {
    return generation.status();
  }
  std::error_code ec;
  const std::uintmax_t bytes =
      std::filesystem::file_size(dir / detail::index_file_name(generation.value()), ec);
  {
    const std::unique_lock<std::shared_mutex> lock(impl_->mutex);
    const auto it = impl_->slots.find(name);
    if (it != impl_->slots.end() && it->second.collection == collection) {
      it->second.generation = generation.value();
    }
  }
  return SnapshotInfo{.generation = generation.value(), .bytes = ec ? 0 : bytes};
}

Status Catalog::snapshot_all() {
  Status first;
  for (const std::string& name : list()) {
    Result<SnapshotInfo> info = snapshot(name);
    if (!info.ok() && info.status().code() != ErrorCode::NotFound && first.ok()) {
      first = info.status();
    }
  }
  return first;
}

const std::filesystem::path& Catalog::data_dir() const noexcept {
  return impl_->data_dir;
}

}  // namespace vf
