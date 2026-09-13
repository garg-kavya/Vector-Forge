// Custom Google Benchmark entry point: records build metadata in the JSON "context" so every
// committed result file is traceable to a commit, build type and compiler.
#include <benchmark/benchmark.h>

#include <string>

#include <vectorforge/version.hpp>

#include "bench_context.hpp"

int main(int argc, char** argv) {
  benchmark::AddCustomContext("vf_version", std::string(vf::kVersion));
  benchmark::AddCustomContext("vf_git_sha", std::string(vf::kGitSha));
  benchmark::AddCustomContext("vf_build_type", std::string(vf::kBuildType));
  benchmark::AddCustomContext("vf_compiler", std::string(vf::kCompiler));
  vf::bench::add_extra_context();

  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
