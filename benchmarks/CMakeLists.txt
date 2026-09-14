add_executable(vf_micro_bench
  micro/bench_main.cpp
  micro/bench_dispatch.cpp
  micro/bench_flat_search.cpp
  micro/bench_kernels.cpp
  micro/bench_topk.cpp
  micro/bench_smoke.cpp
  micro/bench_visited.cpp
)

target_link_libraries(vf_micro_bench PRIVATE vectorforge::vectorforge benchmark::benchmark)
target_include_directories(vf_micro_bench PRIVATE ${PROJECT_SOURCE_DIR}/src)
vf_set_project_options(vf_micro_bench)

# Dispatch-level experiment: the inlined variant lives in its own AVX2-compiled source.
if(VF_AVX2_KERNELS)
  add_library(vf_bench_dispatch_avx2 OBJECT micro/bench_dispatch_avx2.cpp)
  target_include_directories(vf_bench_dispatch_avx2 PRIVATE ${PROJECT_SOURCE_DIR}/src)
  vf_set_project_options(vf_bench_dispatch_avx2)
  vf_avx2_compile_options(vf_bench_avx2_flags)
  target_compile_options(vf_bench_dispatch_avx2 PRIVATE ${vf_bench_avx2_flags})
  target_sources(vf_micro_bench PRIVATE $<TARGET_OBJECTS:vf_bench_dispatch_avx2>)
  target_compile_definitions(vf_micro_bench PRIVATE VF_BENCH_HAVE_AVX2_TU=1)
endif()

# Macro benchmark runner (minimal in Phase 3). Uses CLI11, which is fetched with the CLI.
if(VF_BUILD_CLI)
  add_executable(vf_bench macro/vf_bench.cpp)
  target_link_libraries(vf_bench PRIVATE vectorforge::vectorforge CLI11::CLI11)
  target_include_directories(vf_bench PRIVATE ${PROJECT_SOURCE_DIR}/src)
  vf_set_project_options(vf_bench)
endif()
