add_executable(vf_micro_bench
  micro/bench_main.cpp
  micro/bench_flat_search.cpp
  micro/bench_kernels.cpp
  micro/bench_topk.cpp
  micro/bench_smoke.cpp
)

target_link_libraries(vf_micro_bench PRIVATE vectorforge::vectorforge benchmark::benchmark)
target_include_directories(vf_micro_bench PRIVATE ${PROJECT_SOURCE_DIR}/src)
vf_set_project_options(vf_micro_bench)
