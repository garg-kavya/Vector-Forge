#pragma once

#include <string>

namespace vf::detail {

// Human-readable one-line description of this build (version, git sha, build type, compiler).
[[nodiscard]] std::string build_summary();

}  // namespace vf::detail
