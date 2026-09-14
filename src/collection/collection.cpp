#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <vectorforge/collection.hpp>
#include <vectorforge/version.hpp>

#include "collection/collection_factory.hpp"
#include "collection/collection_state.hpp"
#include "collection/index_file.hpp"
#include "core/assert.hpp"
#include "core/checked_math.hpp"
#include "core/validation.hpp"
#include "core/vector_ops.hpp"
#include "index/flat_backend.hpp"
#include "index/hnsw/hnsw_backend.hpp"
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
// reservation is held. On success consumes the reservation; on any error or exception releases it
// and leaves the collection unchanged (strong guarantee).
Status insert_reserved(CollectionState& state, const IdMap::Reservation& reservation,
                       std::span<const float> row) {
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
    indexed = state.backend->add(internal);
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

}  // namespace
}  // namespace detail

struct Collection::Impl {
  std::unique_ptr<detail::CollectionState> state;
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
  auto impl = std::make_unique<Collection::Impl>();
  impl->state = std::move(state).value();
  return std::unique_ptr<Collection>(new Collection(std::move(impl)));
}

Status CollectionFactory::save_to(const Collection& collection, ByteSink& sink) {
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
      mapping->data(),
      {.verify = verify,
       .owner = options.use_mmap ? std::shared_ptr<const void>(mapping) : nullptr});
  if (!state.ok()) {
    return Status(state.status().code(), file.string() + ": " + state.status().message());
  }
  if (options.use_mmap) {
    mapping->advise(state.value()->config.index == IndexType::Hnsw
                        ? detail::AccessPattern::Random
                        : detail::AccessPattern::Sequential);
  }
  auto impl = std::make_unique<Impl>();
  impl->state = std::move(state).value();
  return std::unique_ptr<Collection>(new Collection(std::move(impl)));
}

Status Collection::save(const std::filesystem::path& file) const {
  const detail::CollectionState& s = *impl_->state;
  return detail::write_atomic(
      file, [&s](detail::ByteSink& sink) { return detail::write_index(s, sink); });
}

Result<std::unique_ptr<Collection>> Collection::create(const CollectionConfig& config) {
  VF_RETURN_IF_ERROR(config.validate());
  Result<detail::VectorStore> store = detail::VectorStore::create({.dim = config.dim});
  if (!store.ok()) {
    return store.status();
  }
  const detail::KernelTable& kernels = detail::kernels();
  auto state = std::make_unique<detail::CollectionState>(config, std::move(store).value(), kernels);
  state->creator = "vectorforge " + std::string(kVersion);
  state->created_unix_ms =
      static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::system_clock::now().time_since_epoch())
                                     .count());
  switch (config.index) {
    case IndexType::Flat:
      state->backend = std::make_unique<detail::FlatBackend>(
          state->vectors, state->deleted, config.metric, state->normalized, kernels);
      break;
    case IndexType::Hnsw: {
      Result<std::unique_ptr<detail::HnswBackend>> hnsw = detail::HnswBackend::create(
          state->vectors, state->deleted, config.metric, state->normalized, kernels, config.hnsw);
      if (!hnsw.ok()) {
        return hnsw.status();
      }
      state->backend = std::move(hnsw).value();
      break;
    }
  }
  VF_CHECK(state->backend != nullptr, "Collection::create: validated index type has no backend");

  auto impl = std::make_unique<Impl>();
  impl->state = std::move(state);
  return std::unique_ptr<Collection>(new Collection(std::move(impl)));
}

Status Collection::add(ExternalId id, std::span<const float> vector, InsertOptions options) {
  detail::CollectionState& s = *impl_->state;
  VF_RETURN_IF_ERROR(detail::validate_vector(vector, s.config.dim));

  std::vector<float> normalized;
  std::span<const float> row = vector;
  if (s.normalized) {
    normalized.assign(vector.begin(), vector.end());
    VF_RETURN_IF_ERROR(detail::normalize_inplace(normalized, *s.kernels));
    row = normalized;
  }

  Result<detail::IdMap::Reservation> reservation = s.ids.reserve(id, options.upsert);
  if (!reservation.ok()) {
    return reservation.status();
  }
  return detail::insert_reserved(s, reservation.value(), row);
}

Result<std::size_t> Collection::add_batch(std::span<const ExternalId> ids,
                                          std::span<const float> rows, InsertOptions options) {
  detail::CollectionState& s = *impl_->state;
  const std::size_t n = ids.size();
  const std::uint32_t dim = s.config.dim;
  VF_RETURN_IF_ERROR(detail::validate_batch(rows, n, dim));
  if (n == 0) {
    return std::size_t{0};
  }

  // Keys: reserved value, duplicates within the batch, existing keys.
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

  // Normalisability of every row is checked before mutating anything.
  if (s.normalized) {
    for (std::size_t i = 0; i < n; ++i) {
      const Result<float> inv = detail::inverse_norm(rows.subspan(i * dim, dim), *s.kernels);
      if (!inv.ok()) {
        return Status::invalid_argument("row " + std::to_string(i) + ": " + inv.status().message());
      }
    }
  }

  VF_RETURN_IF_ERROR(s.vectors.reserve(s.vectors.size() + n));
  s.ids.reserve_capacity(n);
  s.deleted.ensure_size(s.vectors.size() + n);

  std::vector<float> scratch(s.normalized ? dim : 0);
  std::size_t inserted = 0;
  for (std::size_t i = 0; i < n; ++i) {
    std::span<const float> row = rows.subspan(i * dim, dim);
    if (s.normalized) {
      std::copy(row.begin(), row.end(), scratch.begin());
      VF_RETURN_IF_ERROR(detail::normalize_inplace(scratch, *s.kernels));
      row = scratch;
    }
    Result<detail::IdMap::Reservation> reservation = s.ids.reserve(ids[i], options.upsert);
    if (!reservation.ok()) {
      return reservation.status();  // unreachable after the checks above; kept for safety
    }
    const Status st = detail::insert_reserved(s, reservation.value(), row);
    if (!st.ok()) {
      return st;
    }
    ++inserted;
  }
  return inserted;
}

Status Collection::remove(ExternalId id) {
  detail::CollectionState& s = *impl_->state;
  Result<InternalId> internal = s.ids.erase(id);
  if (!internal.ok()) {
    return internal.status();
  }
  s.deleted.set(internal.value());
  s.backend->remove(internal.value());
  return {};
}

Result<std::vector<float>> Collection::get(ExternalId id) const {
  const detail::CollectionState& s = *impl_->state;
  const std::optional<InternalId> internal = s.ids.find(id);
  if (!internal) {
    return Status::not_found("id " + std::to_string(id) + " not found");
  }
  const std::span<const float> row = s.vectors.row(*internal);
  return std::vector<float>(row.begin(), row.end());
}

bool Collection::contains(ExternalId id) const noexcept {
  return impl_->state->ids.contains(id);
}

Result<std::vector<Neighbor>> Collection::search(std::span<const float> query,
                                                 const SearchParams& params) const {
  const detail::CollectionState& s = *impl_->state;
  VF_RETURN_IF_ERROR(params.validate());
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
  const detail::CollectionState& s = *impl_->state;
  VF_RETURN_IF_ERROR(params.validate());
  if (out.size() < params.k) {
    return Status::invalid_argument("output span holds " + std::to_string(out.size()) +
                                    " results but k = " + std::to_string(params.k));
  }
  Result<detail::QueryView> view = detail::prepare_query(s, query);
  if (!view.ok()) {
    return view.status();
  }
  return detail::run_query(s, view.value(), detail::knobs_for(params, s.vectors.size()), out);
}

Status Collection::search_batch(std::span<const float> queries, std::size_t nq,
                                const SearchParams& params, std::span<ExternalId> out_ids,
                                std::span<float> out_distances,
                                std::span<std::uint32_t> counts) const {
  const detail::CollectionState& s = *impl_->state;
  VF_RETURN_IF_ERROR(params.validate());
  const std::uint32_t dim = s.config.dim;
  VF_RETURN_IF_ERROR(detail::validate_batch(queries, nq, dim));
  const auto slots = detail::checked_mul(nq, std::size_t{params.k});
  if (!slots || out_ids.size() < *slots || out_distances.size() < *slots || counts.size() < nq) {
    return Status::invalid_argument("output spans are too small for " + std::to_string(nq) +
                                    " queries with k = " + std::to_string(params.k));
  }
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
  std::vector<Neighbor> scratch(knobs.k);
  for (std::size_t i = 0; i < nq; ++i) {
    const std::size_t count = detail::run_query(s, views[i], knobs, scratch);
    const std::size_t base = i * k;
    for (std::size_t j = 0; j < k; ++j) {
      const bool real = j < count;
      out_ids[base + j] = real ? scratch[j].id : kInvalidExternalId;
      out_distances[base + j] = real ? scratch[j].distance : std::numeric_limits<float>::infinity();
    }
    counts[i] = static_cast<std::uint32_t>(count);
  }
  return {};
}

std::size_t Collection::size() const noexcept {
  return impl_->state->ids.size();
}

CollectionStats Collection::stats() const {
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
  return impl_->state->config;
}

}  // namespace vf
