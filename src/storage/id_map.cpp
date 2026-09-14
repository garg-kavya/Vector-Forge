#include "storage/id_map.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>

#include "core/assert.hpp"

namespace vf::detail {

Result<IdMap::Reservation> IdMap::reserve(ExternalId external, bool upsert) {
  if (external == kInvalidExternalId) {
    return Status::invalid_argument("external id " + std::to_string(external) + " is reserved");
  }
  auto [it, inserted] = map_.try_emplace(external);
  Entry& entry = it->second;
  if (!inserted) {
    if (entry.pending) {
      return Status::already_exists("id " + std::to_string(external) +
                                    " has an insert in progress");
    }
    if (!upsert) {
      return Status::already_exists("id " + std::to_string(external) + " already exists");
    }
  }
  entry.pending = true;
  return Reservation{.external = external, .previous = entry.current};
}

InternalId IdMap::commit(const Reservation& reservation, InternalId internal) noexcept {
  VF_ASSERT(internal == labels_.size(), "IdMap::commit: labels must be appended in row order");
  const auto it = map_.find(reservation.external);
  // Always checked: a commit without a reservation would corrupt the mapping.
  VF_CHECK(it != map_.end() && it->second.pending, "IdMap::commit without reservation");
  Entry& entry = it->second;
  const InternalId replaced = entry.current;
  entry.current = internal;
  entry.pending = false;
  if (replaced == kInvalidInternalId) {
    ++live_;
  }
  labels_.push_back(reservation.external);  // capacity guaranteed by reserve_capacity()
  return replaced;
}

void IdMap::rollback(const Reservation& reservation) noexcept {
  const auto it = map_.find(reservation.external);
  VF_CHECK(it != map_.end() && it->second.pending, "IdMap::rollback without reservation");
  if (reservation.previous == kInvalidInternalId) {
    map_.erase(it);
  } else {
    it->second.pending = false;
  }
}

Result<InternalId> IdMap::erase(ExternalId external) {
  const auto it = map_.find(external);
  if (it == map_.end() || it->second.current == kInvalidInternalId) {
    return Status::not_found("id " + std::to_string(external) + " not found");
  }
  if (it->second.pending) {
    return Status::already_exists("id " + std::to_string(external) + " has an insert in progress");
  }
  const InternalId internal = it->second.current;
  map_.erase(it);
  --live_;
  return internal;
}

std::optional<InternalId> IdMap::find(ExternalId external) const noexcept {
  const auto it = map_.find(external);
  if (it == map_.end() || it->second.current == kInvalidInternalId) {
    return std::nullopt;
  }
  return it->second.current;
}

ExternalId IdMap::label(InternalId internal) const noexcept {
  VF_ASSERT(internal < labels_.size(), "IdMap::label: internal id out of range");
  return labels_[internal];
}

void IdMap::reserve_capacity(std::size_t additional) {
  map_.reserve(map_.size() + additional);
  labels_.reserve(labels_.size() + additional);
}

Result<IdMap> IdMap::restore(std::vector<ExternalId> labels, const TombstoneSet& deleted) {
  IdMap map;
  map.labels_ = std::move(labels);
  const std::size_t rows = map.labels_.size();
  map.map_.reserve(rows - std::min<std::size_t>(rows, static_cast<std::size_t>(deleted.count())));
  for (std::size_t i = 0; i < rows; ++i) {
    const auto internal = static_cast<InternalId>(i);
    if (deleted.test(internal)) {
      continue;
    }
    const ExternalId external = map.labels_[i];
    if (external == kInvalidExternalId) {
      return Status::corrupt_data("LABELS: live row " + std::to_string(i) +
                                  " has the reserved external id");
    }
    const auto [it, inserted] = map.map_.try_emplace(external, Entry{.current = internal});
    if (!inserted) {
      return Status::corrupt_data("LABELS: external id " + std::to_string(external) +
                                  " labels live rows " + std::to_string(it->second.current) +
                                  " and " + std::to_string(i));
    }
  }
  map.live_ = map.map_.size();
  return map;
}

std::size_t IdMap::labels_bytes() const noexcept {
  return labels_.capacity() * sizeof(ExternalId);
}

std::size_t IdMap::map_bytes_estimate() const noexcept {
  // Node: value + next pointer (+ cached hash on some implementations); bucket: one pointer.
  constexpr std::size_t kNodeOverhead = 2 * sizeof(void*);
  return (map_.bucket_count() * sizeof(void*)) +
         (map_.size() * (sizeof(std::pair<const ExternalId, Entry>) + kNodeOverhead));
}

}  // namespace vf::detail
