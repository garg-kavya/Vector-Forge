#pragma once

// Bidirectional mapping between external ids (user keys) and internal ids (row numbers).
//
// external -> internal: hash map; internal -> external: append-only label array indexed by
// internal id (labels of replaced/removed rows are kept so any row can be labelled).
//
// Writes use a two-step protocol so that a failed insert leaves no trace and so that later
// concurrent writers can detect in-flight inserts of the same key (docs/DESIGN.md §11.4):
//   reserve(ext)  -> marks `ext` pending (invisible to find/contains for new keys; an upsert keeps
//                    the old mapping visible until commit)
//   commit(res, internal) -> publishes the mapping and appends the label
//   rollback(res) -> restores the previous state
//
// Thread safety: thread-compatible (Phase 2). Phase 6b adds internal locking.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include <vectorforge/status.hpp>
#include <vectorforge/types.hpp>

namespace vf::detail {

class IdMap {
 public:
  struct Reservation {
    ExternalId external = kInvalidExternalId;
    // Internal id currently mapped to `external`, or kInvalidInternalId for a new key.
    InternalId previous = kInvalidInternalId;
  };

  // Reserves `external` for insertion.
  // Errors: InvalidArgument for kInvalidExternalId; AlreadyExists if the key is mapped and !upsert,
  // or if another reservation for the key is outstanding.
  [[nodiscard]] Result<Reservation> reserve(ExternalId external, bool upsert);

  // Publishes `external -> internal`. `internal` must equal label_count() (labels are appended in
  // row order). Returns the replaced internal id for an upsert, or kInvalidInternalId.
  // Never allocates if reserve_capacity() covered this insert.
  InternalId commit(const Reservation& reservation, InternalId internal) noexcept;

  // Cancels a reservation, restoring the previous mapping.
  void rollback(const Reservation& reservation) noexcept;

  // Cancels a reservation whose row was already appended at `internal`: the label is recorded (so
  // the row stays labelled) but the key mapping is restored. The caller must tombstone `internal`.
  void rollback_appended(const Reservation& reservation, InternalId internal) noexcept;

  // Removes a committed mapping and returns the internal id it pointed to.
  // Errors: NotFound.
  [[nodiscard]] Result<InternalId> erase(ExternalId external);

  [[nodiscard]] std::optional<InternalId> find(ExternalId external) const noexcept;
  [[nodiscard]] bool contains(ExternalId external) const noexcept {
    return find(external).has_value();
  }

  // External id of row `internal`. Precondition: internal < label_count().
  [[nodiscard]] ExternalId label(InternalId internal) const noexcept;

  // Number of visible (committed) mappings.
  [[nodiscard]] std::size_t size() const noexcept { return live_; }
  // Number of labelled rows (== rows appended through commit/rollback_appended).
  [[nodiscard]] std::size_t label_count() const noexcept { return labels_.size(); }

  // Pre-allocates room for `additional` new keys and labels so that commit() cannot allocate.
  void reserve_capacity(std::size_t additional);

  // Approximate heap bytes (labels exact; hash map estimated from bucket and node counts).
  [[nodiscard]] std::size_t labels_bytes() const noexcept;
  [[nodiscard]] std::size_t map_bytes_estimate() const noexcept;

 private:
  struct Entry {
    InternalId current =
        kInvalidInternalId;  // visible mapping; kInvalid while a new key is pending
    bool pending = false;
  };

  std::unordered_map<ExternalId, Entry> map_;
  std::vector<ExternalId> labels_;
  std::size_t live_ = 0;
};

}  // namespace vf::detail
