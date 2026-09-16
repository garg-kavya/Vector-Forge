#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <vectorforge/collection.hpp>
#include <vectorforge/thread_pool.hpp>
#include <vectorforge/version.hpp>

#include "collection/collection_factory.hpp"
#include "collection/collection_state.hpp"
#include "collection/index_file.hpp"
#include "concurrency/fair_shared_mutex.hpp"
#include "core/assert.hpp"
#include "core/checked_math.hpp"
#include "core/validation.hpp"
#include "core/vector_ops.hpp"
#include "index/flat_backend.hpp"
#include "index/hnsw/hnsw_backend.hpp"
#include "simd/dispatch.hpp"
#include "simd/kernels.hpp"
#include "storage/atomic_file.hpp"
#include "storage/mapped_file.hpp"

namespace vf {

namespace detail {
namespace {

// Validates a query and computes its QueryView. Allocation-free on success.
Result<QueryView> prepare_query(const CollectionState& state, std::span<const float> query) {
  VF_RETURN_IF_ERROR(validate_vector(query, state.config.dim));
  QueryView view{.data = query.data(), .inv_norm = 1.0F};
  if (state.normalized) {
    const Result<float> inv = inverse_norm(query, *state.kernels);
    if (!inv.ok()) {
      return Status::invalid_argument("query: " + inv.status().message());
    }
    view.inv_norm = inv.value();
  }
  return view;
}

// Backends need k slots of working storage; k larger than the number of stored rows cannot yield
// more results, so it is clamped to bound temporary allocations in search()/search_batch().
SearchKnobs knobs_for(const SearchParams& params, std::uint64_t rows) noexcept {
  const std::uint64_t k = std::min<std::uint64_t>(params.k, std::max<std::uint64_t>(rows, 1));
  return {.k = static_cast<std::uint32_t>(k), .ef = params.ef_search.value_or(0)};
}

// Runs a validated query and rewrites internal ids to external ids.
std::size_t run_query(const CollectionState& state, const QueryView& view, const SearchKnobs& knobs,
                      std::span<Neighbor> out) {
  const std::size_t count = state.backend->search(view, knobs, out);
  for (std::size_t i = 0; i < count; ++i) {
    out[i].id = state.ids.label(static_cast<InternalId>(out[i].id));
  }
  return count;
}

// Inserts an already validated (and, for normalised collections, already normalised) row whose id
// reservation is held; `index(internal)` indexes the appended row and must itself give the strong
// guarantee. On success consumes the reservation; on any error or exception releases it and leaves
// the collection unchanged (strong guarantee).
template <class IndexFn>
Status insert_reserved(CollectionState& state, const IdMap::Reservation& reservation,
                       std::span<const float> row, IndexFn&& index) {
  // Pre-allocate everything that commit needs so that after indexing nothing can fail.
  try {
    state.ids.reserve_capacity(1);
    state.deleted.ensure_size(state.vectors.size() + 1);
  } catch (...) {
    state.ids.rollback(reservation);
    throw;
  }

  Result<InternalId> appended = [&]() -> Result<InternalId> {
    try {
      return state.vectors.append(row);
    } catch (...) {
      state.ids.rollback(reservation);
      throw;
    }
  }();
  if (!appended.ok()) {
    state.ids.rollback(reservation);
    return appended.status();
  }
  const InternalId internal = appended.value();

  // Backends provide the strong guarantee for add(), so undoing the append keeps row ids and
  // backend node ids aligned (HNSW requires node id == row id).
  Status indexed;
  try {
    indexed = index(internal);
  } catch (...) {
    state.vectors.pop_back();
    state.ids.rollback(reservation);
    throw;
  }
  if (!indexed.ok()) {
    state.vectors.pop_back();
    state.ids.rollback(reservation);
    return indexed;
  }

  const InternalId replaced = state.ids.commit(reservation, internal);
  if (replaced != kInvalidInternalId) {
    state.deleted.set(replaced);
    state.backend->remove(replaced);
  }
  return {};
}

Status insert_reserved(CollectionState& state, const IdMap::Reservation& reservation,
                       std::span<const float> row) {
  return insert_reserved(state, reservation, row,
                         [&state](InternalId internal) { return state.backend->add(internal); });
}

// Creates the configured backend over `state`'s vectors and tombstones.
Status attach_backend(CollectionState& state) {
  switch (state.config.index) {
    case IndexType::Flat:
      state.backend = std::make_unique<FlatBackend>(
          state.vectors, state.deleted, state.config.metric, state.normalized, *state.kernels);
      return {};
    case IndexType::Hnsw: {
      Result<std::unique_ptr<HnswBackend>> hnsw =
          HnswBackend::create(state.vectors, state.deleted, state.config.metric, state.normalized,
                              *state.kernels, state.config.hnsw);
      if (!hnsw.ok()) {
        return hnsw.status();
      }
      state.backend = std::move(hnsw).value();
      return {};
    }
  }
  return Status::invalid_argument("unknown index type");
}

using ReadLock = std::shared_lock<FairSharedMutex>;
using WriteLock = std::unique_lock<FairSharedMutex>;

// Longest exclusive-lock section in add_batch() (at least one row per section): searches queued
// behind a batch run between sections (docs/concurrency.md, "ingest" measurements).
constexpr std::chrono::microseconds kWriteSliceTime{2000};
// Queries per parallel_for chunk in search_batch().
constexpr std::size_t kSearchGrain = 16;
// Rows per parallel_for chunk when normalising a batch.
constexpr std::size_t kNormalizeGrain = 256;
// Level B: rows per link section per participating thread. Searches continue while a section is
// linked; saves and compaction wait for it.
constexpr std::size_t kLinkRowsPerWorker = 64;

// Undoes a row that was committed but could not be linked: the row is tombstoned and the id maps
// to what it mapped to before the insert.
void abandon_row(CollectionState& state, const IdMap::Reservation& reservation,
                 InternalId internal) noexcept {
  state.ids.revert(reservation, internal);
  state.deleted.set(internal);
  if (reservation.previous != kInvalidInternalId) {
    state.deleted.clear(reservation.previous);
  }
}

// Locks and state of a collection, as seen by the Level B batch insert.
struct SyncedState {
  FairSharedMutex& rw;
  std::mutex& link;
  std::shared_ptr<CollectionState>& state;
};

// Level B add_batch (docs/concurrency.md): alternates an exclusive "grow" section, which appends
// rows, allocates their graph nodes and publishes their ids (everything that can fail), with a
// shared "link" section, which connects them while searches continue, on `pool` if given.
// Precondition: the caller holds the collection's writer mutex and validated the batch.
Result<std::size_t> add_batch_concurrent(const SyncedState& sync, std::span<const ExternalId> ids,
                                         std::span<const float> data, std::size_t dim,
                                         InsertOptions options, ThreadPool* pool) {
  const std::size_t n = ids.size();
  const std::size_t workers = pool != nullptr ? pool->size() + 1 : 1;
  const std::size_t section_rows = kLinkRowsPerWorker * workers;
  std::vector<IdMap::Reservation> grown;
  grown.reserve(section_rows);
  std::vector<char> failed(section_rows);

  std::size_t done = 0;
  std::size_t inserted = 0;
  while (done < n) {
    // Saves hold `link` too, so they never see rows between growing and linking.
    const std::lock_guard<std::mutex> link_guard(sync.link);
    Status grow_status;
    std::exception_ptr grow_exception;
    InternalId first = 0;
    grown.clear();
    {
      const WriteLock lock(sync.rw);
      CollectionState& s = *sync.state;
      auto& hnsw = static_cast<HnswBackend&>(*s.backend);
      const std::size_t want = std::min(section_rows, n - done);
      hnsw.prepare_insert(s.vectors.size() + want, workers);  // may throw: nothing changed yet
      first = static_cast<InternalId>(s.vectors.size());
      const auto slice_end = std::chrono::steady_clock::now() + kWriteSliceTime;
      try {
        while (grown.size() < want) {
          const std::size_t row = done + grown.size();
          Result<IdMap::Reservation> reservation = s.ids.reserve(ids[row], options.upsert);
          if (!reservation.ok()) {
            grow_status = reservation.status();  // unreachable after validation; kept for safety
            break;
          }
          grow_status =
              insert_reserved(s, reservation.value(), data.subspan(row * dim, dim),
                              [&hnsw](InternalId internal) { return hnsw.reserve_node(internal); });
          if (!grow_status.ok()) {
            break;
          }
          grown.push_back(reservation.value());  // capacity reserved above
          if (std::chrono::steady_clock::now() >= slice_end) {
            break;
          }
        }
      } catch (...) {
        grow_exception = std::current_exception();  // the grown rows are linked first
      }
    }

    std::atomic<bool> link_failed{false};
    if (!grown.empty()) {
      const ReadLock lock(sync.rw);
      auto& hnsw = static_cast<HnswBackend&>(*sync.state->backend);
      auto link_range = [&](std::size_t lo, std::size_t hi) {
        for (std::size_t i = lo; i < hi; ++i) {
          failed[i] = 0;
          try {
            hnsw.link(first + static_cast<InternalId>(i));
          } catch (const std::bad_alloc&) {
            failed[i] = 1;
            link_failed.store(true);
          }
        }
      };
      if (pool != nullptr && grown.size() > 1) {
        pool->parallel_for(0, grown.size(), 1, link_range);
      } else {
        link_range(0, grown.size());
      }
    }

    std::size_t linked = grown.size();
    if (link_failed.load()) {
      const WriteLock lock(sync.rw);
      for (std::size_t i = 0; i < grown.size(); ++i) {
        if (failed[i] != 0) {
          abandon_row(*sync.state, grown[i], first + static_cast<InternalId>(i));
          --linked;
        }
      }
    }
    done += grown.size();
    inserted += linked;
    if (grow_exception) {
      std::rethrow_exception(grow_exception);
    }
    if (link_failed.load()) {
      throw std::bad_alloc();
    }
    if (!grow_status.ok()) {
      return grow_status;
    }
  }
  return inserted;
}

}  // namespace
}  // namespace detail

struct Collection::Impl {
  explicit Impl(std::shared_ptr<detail::CollectionState> initial)
      : config(initial->config), state(std::move(initial)) {}

  const CollectionConfig config;  // immutable copy: config() needs no lock
  // Synchronisation (docs/concurrency.md). Lock order: writers, link, rw.
  // Readers hold `rw` shared; mutations hold `writers` and then `rw` exclusively; compact() holds
  // `writers` and `rw` shared while it builds, then `rw` exclusively for the swap. Level B inserts
  // hold `writers`, and `link` for each grow-and-link section (taking `rw` exclusively to grow,
  // shared to link); saves hold `link` so that they never see grown but unlinked rows.
  mutable detail::FairSharedMutex rw;
  std::mutex writers;
  mutable std::mutex link;
  std::shared_ptr<detail::CollectionState> state;
};

Collection::Collection(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {
}
Collection::~Collection() = default;

namespace detail {

Result<std::unique_ptr<Collection>> CollectionFactory::load_from_memory(
    std::span<const std::byte> file, Verify verify) {
  Result<std::unique_ptr<CollectionState>> state = read_index(
      file, {.verify = verify == Verify::Auto ? Verify::Full : verify, .owner = nullptr});
  if (!state.ok()) {
    return state.status();
  }
  auto impl = std::make_unique<Collection::Impl>(std::move(state).value());
  return std::unique_ptr<Collection>(new Collection(std::move(impl)));
}

Status CollectionFactory::save_to(const Collection& collection, ByteSink& sink) {
  const std::lock_guard<std::mutex> link(collection.impl_->link);
  const detail::ReadLock lock(collection.impl_->rw);
  return write_index(*collection.impl_->state, sink);
}

const CollectionState& CollectionFactory::state(const Collection& collection) noexcept {
  return *collection.impl_->state;
}

}  // namespace detail

Result<std::unique_ptr<Collection>> Collection::load(const std::filesystem::path& file,
                                                     const LoadOptions& options) {
  Result<detail::MappedFile> mapped = detail::MappedFile::open(file);
  if (!mapped.ok()) {
    return mapped.status();
  }
  auto mapping = std::make_shared<detail::MappedFile>(std::move(mapped).value());
  Verify verify = options.verify;
  if (verify == Verify::Auto) {
    verify = options.use_mmap ? Verify::Metadata : Verify::Full;
  }
  if (options.use_mmap && options.prefault) {
    mapping->prefault();
  }
  // Heap loads also parse through the mapping (page cache instead of a private copy of the whole
  // file); the mapping is released when read_index returns because nothing keeps it.
  Result<std::unique_ptr<detail::CollectionState>> state = detail::read_index(
      mapping->data(), {.verify = verify,
                        .owner = options.use_mmap ? std::shared_ptr<const void>(mapping) : nullptr,
                        .concurrency = options.concurrency});
  if (!state.ok()) {
    return Status(state.status().code(), file.string() + ": " + state.status().message());
  }
  if (options.use_mmap) {
    mapping->advise(state.value()->config.index == IndexType::Hnsw
                        ? detail::AccessPattern::Random
                        : detail::AccessPattern::Sequential);
  }
  auto impl = std::make_unique<Impl>(std::move(state).value());
  return std::unique_ptr<Collection>(new Collection(std::move(impl)));
}

Status Collection::save(const std::filesystem::path& file) const {
  const std::lock_guard<std::mutex> link(impl_->link);
  const detail::ReadLock lock(impl_->rw);
  const detail::CollectionState& s = *impl_->state;
  return detail::write_atomic(
      file, [&s](detail::ByteSink& sink) { return detail::write_index(s, sink); });
}

Result<std::unique_ptr<Collection>> Collection::create(const CollectionConfig& config) {
  VF_RETURN_IF_ERROR(config.validate());
  VF_RETURN_IF_ERROR(detail::kernel_selection().status);
  Result<detail::VectorStore> store = detail::VectorStore::create({.dim = config.dim});
  if (!store.ok()) {
    return store.status();
  }
  auto state = std::make_unique<detail::CollectionState>(config, std::move(store).value(),
                                                         detail::kernels());
  state->creator = "vectorforge " + std::string(kVersion);
  state->created_unix_ms =
      static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::system_clock::now().time_since_epoch())
                                     .count());
  VF_RETURN_IF_ERROR(detail::attach_backend(*state));
  VF_CHECK(state->backend != nullptr, "Collection::create: validated index type has no backend");

  auto impl = std::make_unique<Impl>(std::move(state));
  return std::unique_ptr<Collection>(new Collection(std::move(impl)));
}

Status Collection::add(ExternalId id, std::span<const float> vector, InsertOptions options) {
  VF_RETURN_IF_ERROR(detail::validate_vector(vector, impl_->config.dim));

  std::vector<float> normalized;
  std::span<const float> row = vector;
  if (impl_->config.effective_normalize()) {
    normalized.assign(vector.begin(), vector.end());
    VF_RETURN_IF_ERROR(detail::normalize_inplace(normalized, detail::kernels()));
    row = normalized;
  }

  const std::lock_guard<std::mutex> writer(impl_->writers);
  const detail::WriteLock lock(impl_->rw);
  detail::CollectionState& s = *impl_->state;
  Result<detail::IdMap::Reservation> reservation = s.ids.reserve(id, options.upsert);
  if (!reservation.ok()) {
    return reservation.status();
  }
  return detail::insert_reserved(s, reservation.value(), row);
}

Result<std::size_t> Collection::add_batch(std::span<const ExternalId> ids,
                                          std::span<const float> rows, InsertOptions options,
                                          ThreadPool* pool) {
  const std::size_t n = ids.size();
  const std::uint32_t dim = impl_->config.dim;
  const bool normalize = impl_->config.effective_normalize();
  VF_RETURN_IF_ERROR(detail::validate_batch(rows, n, dim));
  if (n == 0) {
    return std::size_t{0};
  }

  // Keys: reserved value, duplicates within the batch.
  std::vector<ExternalId> sorted(ids.begin(), ids.end());
  std::sort(sorted.begin(), sorted.end());
  const auto dup = std::adjacent_find(sorted.begin(), sorted.end());
  if (dup != sorted.end()) {
    return Status::invalid_argument("id " + std::to_string(*dup) +
                                    " appears more than once in batch");
  }
  if (sorted.back() == kInvalidExternalId) {
    return Status::invalid_argument("external id " + std::to_string(kInvalidExternalId) +
                                    " is reserved");
  }

  // Normalised copies are computed (and every row checked for normalisability) before locking.
  std::vector<float> normalized_rows;
  if (normalize) {
    normalized_rows.assign(rows.begin(), rows.end());
    const detail::KernelTable& kernels = detail::kernels();
    std::vector<Status> failures(n);
    auto normalize_range = [&](std::size_t lo, std::size_t hi) {
      for (std::size_t i = lo; i < hi; ++i) {
        failures[i] = detail::normalize_inplace(
            std::span<float>(normalized_rows).subspan(i * dim, dim), kernels);
      }
    };
    if (pool != nullptr) {
      pool->parallel_for(0, n, detail::kNormalizeGrain, normalize_range);
    } else {
      normalize_range(0, n);
    }
    for (std::size_t i = 0; i < n; ++i) {
      if (!failures[i].ok()) {
        return Status::invalid_argument("row " + std::to_string(i) + ": " + failures[i].message());
      }
    }
  }
  const std::span<const float> data = normalize ? std::span<const float>(normalized_rows) : rows;

  // Holding `writers` for the whole batch keeps the validation below valid while the exclusive
  // lock is released between slices so that searches can run.
  const std::lock_guard<std::mutex> writer(impl_->writers);
  {
    const detail::WriteLock lock(impl_->rw);
    detail::CollectionState& s = *impl_->state;
    if (!options.upsert) {
      for (const ExternalId id : ids) {
        if (s.ids.contains(id)) {
          return Status::already_exists("id " + std::to_string(id) + " already exists");
        }
      }
    }
    if (s.vectors.size() + n > s.vectors.max_rows()) {
      return Status::resource_exhausted("batch of " + std::to_string(n) +
                                        " vectors exceeds collection capacity");
    }
    VF_RETURN_IF_ERROR(s.vectors.reserve(s.vectors.size() + n));
    s.ids.reserve_capacity(n);
    s.deleted.ensure_size(s.vectors.size() + n);
  }

  if (impl_->config.index == IndexType::Hnsw &&
      impl_->config.concurrency == Concurrency::Concurrent) {
    return detail::add_batch_concurrent(
        {.rw = impl_->rw, .link = impl_->link, .state = impl_->state}, ids, data, dim, options,
        pool);
  }

  std::size_t inserted = 0;
  while (inserted < n) {
    const detail::WriteLock lock(impl_->rw);
    detail::CollectionState& s = *impl_->state;
    const auto slice_end = std::chrono::steady_clock::now() + detail::kWriteSliceTime;
    do {
      Result<detail::IdMap::Reservation> reservation = s.ids.reserve(ids[inserted], options.upsert);
      if (!reservation.ok()) {
        return reservation.status();  // unreachable after the checks above; kept for safety
      }
      const Status st =
          detail::insert_reserved(s, reservation.value(), data.subspan(inserted * dim, dim));
      if (!st.ok()) {
        return st;
      }
      ++inserted;
    } while (inserted < n && std::chrono::steady_clock::now() < slice_end);
  }
  return inserted;
}

Status Collection::remove(ExternalId id) {
  const std::lock_guard<std::mutex> writer(impl_->writers);
  const detail::WriteLock lock(impl_->rw);
  detail::CollectionState& s = *impl_->state;
  Result<InternalId> internal = s.ids.erase(id);
  if (!internal.ok()) {
    return internal.status();
  }
  s.deleted.set(internal.value());
  s.backend->remove(internal.value());
  return {};
}

Result<CompactStats> Collection::compact() {
  const std::lock_guard<std::mutex> writer(impl_->writers);
  std::unique_ptr<detail::CollectionState> rebuilt;
  CompactStats stats;
  {
    // Searches continue while the new state is built; writers wait on `writers`.
    const detail::ReadLock lock(impl_->rw);
    const detail::CollectionState& old = *impl_->state;
    stats.rows_before = old.vectors.size();
    stats.removed_rows = old.deleted.count();
    stats.rows_after = stats.rows_before;
    if (stats.removed_rows == 0) {
      return stats;
    }
    Result<detail::VectorStore> store = detail::VectorStore::create({.dim = old.config.dim});
    if (!store.ok()) {
      return store.status();
    }
    rebuilt = std::make_unique<detail::CollectionState>(old.config, std::move(store).value(),
                                                        *old.kernels);
    rebuilt->creator = old.creator;
    rebuilt->created_unix_ms = old.created_unix_ms;
    VF_RETURN_IF_ERROR(detail::attach_backend(*rebuilt));
    const std::uint64_t live = old.ids.size();
    VF_RETURN_IF_ERROR(rebuilt->vectors.reserve(live));
    rebuilt->ids.reserve_capacity(live);
    rebuilt->deleted.ensure_size(live);
    // Live rows are re-inserted in their original order; stored rows are already normalised.
    for (std::uint64_t r = 0; r < old.vectors.size(); ++r) {
      const auto internal = static_cast<InternalId>(r);
      if (old.deleted.test(internal)) {
        continue;
      }
      Result<detail::IdMap::Reservation> reservation =
          rebuilt->ids.reserve(old.ids.label(internal), false);
      if (!reservation.ok()) {
        return reservation.status();
      }
      VF_RETURN_IF_ERROR(
          detail::insert_reserved(*rebuilt, reservation.value(), old.vectors.row(internal)));
    }
    stats.rows_after = rebuilt->vectors.size();
  }
  std::shared_ptr<detail::CollectionState> retired;
  {
    const detail::WriteLock lock(impl_->rw);
    retired = std::exchange(impl_->state, std::move(rebuilt));
  }
  // The old state (and any file mapping it holds) is released here, outside the lock.
  return stats;
}

Result<std::vector<float>> Collection::get(ExternalId id) const {
  const detail::ReadLock lock(impl_->rw);
  const detail::CollectionState& s = *impl_->state;
  const std::optional<InternalId> internal = s.ids.find(id);
  if (!internal) {
    return Status::not_found("id " + std::to_string(id) + " not found");
  }
  const std::span<const float> row = s.vectors.row(*internal);
  return std::vector<float>(row.begin(), row.end());
}

bool Collection::contains(ExternalId id) const {
  const detail::ReadLock lock(impl_->rw);
  return impl_->state->ids.contains(id);
}

Result<std::vector<Neighbor>> Collection::search(std::span<const float> query,
                                                 const SearchParams& params) const {
  VF_RETURN_IF_ERROR(params.validate());
  const detail::ReadLock lock(impl_->rw);
  const detail::CollectionState& s = *impl_->state;
  Result<detail::QueryView> view = detail::prepare_query(s, query);
  if (!view.ok()) {
    return view.status();
  }
  const detail::SearchKnobs knobs = detail::knobs_for(params, s.vectors.size());
  std::vector<Neighbor> out(knobs.k);
  const std::size_t count = detail::run_query(s, view.value(), knobs, out);
  out.resize(count);
  return out;
}

Result<std::size_t> Collection::search_into(std::span<const float> query,
                                            const SearchParams& params,
                                            std::span<Neighbor> out) const {
  VF_RETURN_IF_ERROR(params.validate());
  if (out.size() < params.k) {
    return Status::invalid_argument("output span holds " + std::to_string(out.size()) +
                                    " results but k = " + std::to_string(params.k));
  }
  const detail::ReadLock lock(impl_->rw);
  const detail::CollectionState& s = *impl_->state;
  Result<detail::QueryView> view = detail::prepare_query(s, query);
  if (!view.ok()) {
    return view.status();
  }
  return detail::run_query(s, view.value(), detail::knobs_for(params, s.vectors.size()), out);
}

Status Collection::search_batch(std::span<const float> queries, std::size_t nq,
                                const SearchParams& params, std::span<ExternalId> out_ids,
                                std::span<float> out_distances, std::span<std::uint32_t> counts,
                                ThreadPool* pool) const {
  VF_RETURN_IF_ERROR(params.validate());
  const std::uint32_t dim = impl_->config.dim;
  VF_RETURN_IF_ERROR(detail::validate_batch(queries, nq, dim));
  const auto slots = detail::checked_mul(nq, std::size_t{params.k});
  if (!slots || out_ids.size() < *slots || out_distances.size() < *slots || counts.size() < nq) {
    return Status::invalid_argument("output spans are too small for " + std::to_string(nq) +
                                    " queries with k = " + std::to_string(params.k));
  }
  const detail::ReadLock lock(impl_->rw);
  const detail::CollectionState& s = *impl_->state;
  std::vector<detail::QueryView> views;
  views.reserve(nq);
  for (std::size_t i = 0; i < nq; ++i) {
    Result<detail::QueryView> view = detail::prepare_query(s, queries.subspan(i * dim, dim));
    if (!view.ok()) {
      return {view.status().code(), "query " + std::to_string(i) + ": " + view.status().message()};
    }
    views.push_back(view.value());
  }

  const detail::SearchKnobs knobs = detail::knobs_for(params, s.vectors.size());
  const std::size_t k = params.k;
  // Pool workers read the state under this thread's shared lock, which parallel_for keeps held
  // until every chunk has finished.
  auto run_range = [&](std::size_t lo, std::size_t hi) {
    std::vector<Neighbor> scratch(knobs.k);
    for (std::size_t i = lo; i < hi; ++i) {
      const std::size_t count = detail::run_query(s, views[i], knobs, scratch);
      const std::size_t base = i * k;
      for (std::size_t j = 0; j < k; ++j) {
        const bool real = j < count;
        out_ids[base + j] = real ? scratch[j].id : kInvalidExternalId;
        out_distances[base + j] =
            real ? scratch[j].distance : std::numeric_limits<float>::infinity();
      }
      counts[i] = static_cast<std::uint32_t>(count);
    }
  };
  if (pool != nullptr) {
    pool->parallel_for(0, nq, detail::kSearchGrain, run_range);
  } else {
    run_range(0, nq);
  }
  return {};
}

std::size_t Collection::size() const {
  const detail::ReadLock lock(impl_->rw);
  return impl_->state->ids.size();
}

CollectionStats Collection::stats() const {
  const detail::ReadLock lock(impl_->rw);
  const detail::CollectionState& s = *impl_->state;
  CollectionStats st;
  st.live_count = s.ids.size();
  st.deleted_count = s.deleted.count();
  st.row_count = s.vectors.size();
  st.dim = s.config.dim;
  st.metric = s.config.metric;
  st.index = s.config.index;
  st.normalized = s.normalized;
  st.simd = s.kernels->level;
  st.memory.vectors_bytes = s.vectors.allocated_bytes();
  st.memory.mapped_vectors_bytes = s.vectors.mapped_bytes();
  st.memory.labels_bytes = s.ids.labels_bytes();
  st.memory.id_map_bytes_estimate = s.ids.map_bytes_estimate();
  st.memory.tombstone_bytes = s.deleted.bytes();
  st.memory.index_bytes = s.backend->stats().index_bytes;
  return st;
}

const CollectionConfig& Collection::config() const noexcept {
  return impl_->config;
}

}  // namespace vf
