#include "core/build_info.hpp"

#include <string>

#include <vectorforge/version.hpp>

namespace vf::detail {

std::string build_summary() {
  std::string out = "VectorForge ";
  out += kVersion;
  out += " (";
  out += kGitSha;
  out += ", ";
  out += kBuildType;
  out += ", ";
  out += kCompiler;
  out += ")";
  return out;
}

}  // namespace vf::detail
