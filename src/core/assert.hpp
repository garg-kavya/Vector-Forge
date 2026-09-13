#pragma once

// Internal assertion macros.
//
// VF_CHECK(cond, msg)  - always enabled; for invariants whose violation must never go unnoticed.
// VF_ASSERT(cond, msg) - enabled in debug builds (or with VF_ENABLE_ASSERTS); for hot-path
//                        preconditions that are guaranteed by validation at the API boundary.

#include <string_view>

namespace vf::detail {

// Prints "file:line: message" to stderr and aborts.
[[noreturn]] void fatal(const char* file, int line, std::string_view message) noexcept;

}  // namespace vf::detail

// NOLINTBEGIN(cppcoreguidelines-macro-usage)
#define VF_CHECK(cond, msg) \
  ((cond) ? static_cast<void>(0) : ::vf::detail::fatal(__FILE__, __LINE__, (msg)))

#if !defined(NDEBUG) || defined(VF_ENABLE_ASSERTS)
#define VF_ASSERT(cond, msg) VF_CHECK(cond, msg)
#else
#define VF_ASSERT(cond, msg) static_cast<void>(0)
#endif
// NOLINTEND(cppcoreguidelines-macro-usage)
