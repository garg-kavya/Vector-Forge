#pragma once

// Current process memory for benchmarks (docs/DESIGN.md §16.4).
//   rss_bytes        resident set / working set (includes mapped file pages that are resident)
//   private_bytes    Windows: PrivateUsage (commit charge); Linux: RssAnon (anonymous resident)

#include <cstdint>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
// windows.h must precede psapi.h
#include <psapi.h>
#else
#include <fstream>
#include <sstream>
#include <string>
#endif

namespace vf::bench {

struct ProcessMemory {
  std::uint64_t rss_bytes = 0;
  std::uint64_t private_bytes = 0;
  std::uint64_t peak_rss_bytes = 0;  // Windows: PeakWorkingSetSize; Linux: VmHWM
};

inline ProcessMemory process_memory() {
  ProcessMemory m;
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS_EX counters{};
  counters.cb = sizeof(counters);
  if (K32GetProcessMemoryInfo(GetCurrentProcess(),
                              reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                              counters.cb) != 0) {
    m.rss_bytes = counters.WorkingSetSize;
    m.private_bytes = counters.PrivateUsage;
    m.peak_rss_bytes = counters.PeakWorkingSetSize;
  }
#else
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    std::istringstream fields(line);
    std::string key;
    std::uint64_t kib = 0;
    fields >> key >> kib;
    if (key == "VmRSS:") {
      m.rss_bytes = kib * 1024;
    } else if (key == "RssAnon:") {
      m.private_bytes = kib * 1024;
    } else if (key == "VmHWM:") {
      m.peak_rss_bytes = kib * 1024;
    }
  }
#endif
  return m;
}

}  // namespace vf::bench
