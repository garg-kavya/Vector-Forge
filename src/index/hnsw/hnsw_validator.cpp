#include "index/hnsw/hnsw_validator.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

namespace vf::detail {

namespace {

std::string where(InternalId id, std::uint8_t level) {
  return "node " + std::to_string(id) + " level " + std::to_string(level) + ": ";
}

}  // namespace

Status HnswValidator::check_invariants() const {
  const HnswGraph& g = *graph_;
  const std::uint64_t n = g.node_count();
  const HnswGraph::Entry entry = g.entry();
  if (n == 0) {
    return entry.valid() ? Status::internal("empty graph has an entry point") : Status{};
  }
  if (!entry.valid() || entry.id >= n) {
    return Status::internal("non-empty graph has no valid entry point");
  }
  if (g.level(entry.id) != entry.level) {
    return Status::internal("entry level " + std::to_string(entry.level) + " != level of node " +
                            std::to_string(entry.id));
  }

  std::vector<InternalId> sorted;
  sorted.reserve(g.capacity(0));
  std::uint8_t top = 0;
  for (std::uint64_t i = 0; i < n; ++i) {
    const auto id = static_cast<InternalId>(i);
    const std::uint8_t node_level = g.level(id);
    if (node_level > g.max_level()) {
      return Status::internal(where(id, node_level) + "level exceeds max_level");
    }
    top = std::max(top, node_level);
    for (std::uint8_t l = 0; l <= node_level; ++l) {
      const LinkView links = g.links(id, l);
      if (links.size() > g.capacity(l)) {
        return Status::internal(where(id, l) + "count " + std::to_string(links.size()) +
                                " exceeds capacity " + std::to_string(g.capacity(l)));
      }
      sorted.assign(links.ids().begin(), links.ids().end());
      for (const InternalId neighbor : sorted) {
        if (neighbor >= n) {
          return Status::internal(where(id, l) + "link to nonexistent node " +
                                  std::to_string(neighbor));
        }
        if (neighbor == id) {
          return Status::internal(where(id, l) + "self-link");
        }
        if (g.level(neighbor) < l) {
          return Status::internal(where(id, l) + "link to node " + std::to_string(neighbor) +
                                  " whose level is " + std::to_string(g.level(neighbor)));
        }
      }
      std::sort(sorted.begin(), sorted.end());
      const auto dup = std::adjacent_find(sorted.begin(), sorted.end());
      if (dup != sorted.end()) {
        return Status::internal(where(id, l) + "duplicate link to " + std::to_string(*dup));
      }
    }
  }
  if (entry.level != top) {
    return Status::internal("entry level " + std::to_string(entry.level) +
                            " != highest node level " + std::to_string(top));
  }
  return {};
}

HnswReachability HnswValidator::reachability() const {
  const HnswGraph& g = *graph_;
  HnswReachability report;
  report.nodes = g.node_count();
  const HnswGraph::Entry entry = g.entry();
  if (report.nodes == 0 || !entry.valid()) {
    return report;
  }
  const auto n = static_cast<std::size_t>(report.nodes);
  std::vector<char> seen(n);
  std::vector<InternalId> queue;
  queue.reserve(n);
  report.unreachable.assign(static_cast<std::size_t>(entry.level) + 1, 0);
  for (int li = entry.level; li >= 0; --li) {
    const auto l = static_cast<std::uint8_t>(li);
    std::fill(seen.begin(), seen.end(), char{0});
    queue.clear();
    queue.push_back(entry.id);
    seen[entry.id] = 1;
    for (std::size_t head = 0; head < queue.size(); ++head) {
      const LinkView links = g.links(queue[head], l);
      for (const InternalId neighbor : links.ids()) {
        if (seen[neighbor] == 0) {
          seen[neighbor] = 1;
          queue.push_back(neighbor);
        }
      }
    }
    std::uint64_t on_level = 0;
    for (std::size_t i = 0; i < n; ++i) {
      if (g.level(static_cast<InternalId>(i)) >= l) {
        ++on_level;
      }
    }
    report.unreachable[l] = on_level - queue.size();
  }
  return report;
}

std::vector<std::uint64_t> HnswValidator::level_histogram() const {
  const HnswGraph& g = *graph_;
  std::vector<std::uint64_t> histogram(static_cast<std::size_t>(g.max_level()) + 1, 0);
  for (std::uint64_t i = 0; i < g.node_count(); ++i) {
    ++histogram[g.level(static_cast<InternalId>(i))];
  }
  return histogram;
}

}  // namespace vf::detail
