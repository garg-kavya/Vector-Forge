include(GoogleTest)

# Unit tests ------------------------------------------------------------------------------------
add_executable(vf_unit_tests
  unit/test_aligned_alloc.cpp
  unit/test_binary_io.cpp
  unit/test_checked_math.cpp
  unit/test_collection_flat.cpp
  unit/test_compact.cpp
  unit/test_cpu_features.cpp
  unit/test_crc32c.cpp
  unit/test_dataset_io.cpp
  unit/test_dispatch.cpp
  unit/test_flat_backend.cpp
  unit/test_hnsw_determinism.cpp
  unit/test_hnsw_edge_cases.cpp
  unit/test_hnsw_graph.cpp
  unit/test_hnsw_tombstones.cpp
  unit/test_hnsw_validator.cpp
  unit/test_id_map.cpp
  unit/test_kernels.cpp
  unit/test_level_generator.cpp
  unit/test_mapped_file.cpp
  unit/test_model_flat.cpp
  unit/test_neighbor_select.cpp
  unit/test_normalize.cpp
  unit/test_recall.cpp
  unit/test_rng.cpp
  unit/test_smoke.cpp
  unit/test_status.cpp
  unit/test_synthetic.cpp
  unit/test_topk.cpp
  unit/test_types_config.cpp
  unit/test_validation.cpp
  unit/test_vector_store.cpp
  unit/test_visited_set.cpp
)
target_link_libraries(vf_unit_tests PRIVATE vectorforge::vectorforge GTest::gtest_main)
# Tests may exercise internal components directly.
target_include_directories(vf_unit_tests PRIVATE ${PROJECT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
vf_set_project_options(vf_unit_tests)

# PRE_TEST discovery: the test binary is not executed at build time (keeps sanitizer
# runtime DLL lookup and cross-environment builds simple).
gtest_discover_tests(vf_unit_tests
  DISCOVERY_MODE PRE_TEST
  PROPERTIES LABELS unit)

# Kernel tier selection under explicit VF_SIMD requests (docs/simd.md). The default run above uses the
# best tier; these re-run the tier-sensitive tests with the request fixed. The tests are
# environment-aware, so an unsupported or invalid request must be reported, not crash or downgrade.
add_test(NAME simd.forced_scalar
         COMMAND vf_unit_tests --gtest_filter=Dispatch.*:Kernels.*:Normalize.*:HnswDeterminism.*:FlatBackend.*)
set_tests_properties(simd.forced_scalar PROPERTIES ENVIRONMENT "VF_SIMD=scalar" LABELS unit)
add_test(NAME simd.explicit_avx2
         COMMAND vf_unit_tests --gtest_filter=Dispatch.*:Kernels.*:HnswDeterminism.*)
set_tests_properties(simd.explicit_avx2 PROPERTIES ENVIRONMENT "VF_SIMD=avx2" LABELS unit)
add_test(NAME simd.invalid_request COMMAND vf_unit_tests --gtest_filter=Dispatch.*)
set_tests_properties(simd.invalid_request PROPERTIES ENVIRONMENT "VF_SIMD=fastest" LABELS unit)

# Allocation tests: separate binary because it replaces the global operator new/delete -----------
# ThreadSanitizer's runtime defines the global allocation functions itself, so the replacement
# cannot link there; these tests run in every other configuration.
if(NOT VF_SANITIZE MATCHES "thread")
  add_executable(vf_alloc_tests
    alloc/counting_allocator.cpp
    alloc/test_exception_safety.cpp
    alloc/test_zero_alloc_search.cpp
  )
  target_link_libraries(vf_alloc_tests PRIVATE vectorforge::vectorforge GTest::gtest_main)
  target_include_directories(vf_alloc_tests PRIVATE ${PROJECT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
  vf_set_project_options(vf_alloc_tests)
  gtest_discover_tests(vf_alloc_tests
    DISCOVERY_MODE PRE_TEST
    PROPERTIES LABELS unit)
endif()

# Concurrency tests (meaningful under TSan: linux-clang-tsan preset) ---------------------------
find_package(Threads REQUIRED)
add_executable(vf_concurrency_tests
  concurrency/test_concurrent_reads.cpp
  concurrency/test_concurrent_search.cpp
  concurrency/test_thread_pool.cpp
)
target_link_libraries(vf_concurrency_tests
  PRIVATE vectorforge::vectorforge GTest::gtest_main Threads::Threads)
target_include_directories(vf_concurrency_tests
  PRIVATE ${PROJECT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
vf_set_project_options(vf_concurrency_tests)
gtest_discover_tests(vf_concurrency_tests
  DISCOVERY_MODE PRE_TEST
  PROPERTIES LABELS concurrency)

# Stress tests: part of every run; the nightly workflow repeats them (ctest -L stress --repeat).
add_executable(vf_stress_tests
  concurrency/test_rw_stress.cpp
)
target_link_libraries(vf_stress_tests
  PRIVATE vectorforge::vectorforge GTest::gtest_main Threads::Threads)
target_include_directories(vf_stress_tests
  PRIVATE ${PROJECT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
vf_set_project_options(vf_stress_tests)
gtest_discover_tests(vf_stress_tests
  DISCOVERY_MODE PRE_TEST
  PROPERTIES LABELS "concurrency;stress")

# Persistence tests ---------------------------------------------------------------------------
add_executable(vf_persistence_tests
  persistence/test_atomic_save.cpp
  persistence/test_corruption.cpp
  persistence/test_golden_files.cpp
  persistence/test_persistence_roundtrip.cpp
)
target_link_libraries(vf_persistence_tests PRIVATE vectorforge::vectorforge GTest::gtest_main)
target_include_directories(vf_persistence_tests
  PRIVATE ${PROJECT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(vf_persistence_tests PRIVATE VF_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
vf_set_project_options(vf_persistence_tests)
gtest_discover_tests(vf_persistence_tests
  DISCOVERY_MODE PRE_TEST
  PROPERTIES LABELS persistence)

# Index reader fuzz target: corpus replay everywhere, libFuzzer binary with VF_BUILD_FUZZERS ----
add_library(vf_fuzz_index_reader OBJECT fuzz/fuzz_index_reader.cpp)
target_link_libraries(vf_fuzz_index_reader PRIVATE vectorforge::vectorforge)
target_include_directories(vf_fuzz_index_reader PRIVATE ${PROJECT_SOURCE_DIR}/src)
vf_set_project_options(vf_fuzz_index_reader)

add_executable(vf_fuzz_replay fuzz/replay_main.cpp $<TARGET_OBJECTS:vf_fuzz_index_reader>)
target_link_libraries(vf_fuzz_replay PRIVATE vectorforge::vectorforge)
vf_set_project_options(vf_fuzz_replay)
add_test(NAME fuzz.index_reader_replay
         COMMAND vf_fuzz_replay ${CMAKE_CURRENT_SOURCE_DIR}/data/golden)
set_tests_properties(fuzz.index_reader_replay PROPERTIES LABELS persistence)
add_test(NAME simd.load_reports_invalid_request
         COMMAND vf_persistence_tests --gtest_filter=GoldenFiles.ReportsSimdSelectionError)
set_tests_properties(simd.load_reports_invalid_request
                     PROPERTIES ENVIRONMENT "VF_SIMD=fastest" LABELS persistence)

if(VF_BUILD_FUZZERS)
  add_executable(vf_fuzz_index_reader_libfuzzer $<TARGET_OBJECTS:vf_fuzz_index_reader>)
  target_link_libraries(vf_fuzz_index_reader_libfuzzer PRIVATE vectorforge::vectorforge)
  target_link_options(vf_fuzz_index_reader_libfuzzer PRIVATE -fsanitize=fuzzer)
endif()

# Integration tests ---------------------------------------------------------------------------
add_executable(vf_index_integration_tests
  integration/test_hnsw_recall.cpp
)
target_link_libraries(vf_index_integration_tests PRIVATE vectorforge::vectorforge GTest::gtest_main)
target_include_directories(vf_index_integration_tests
  PRIVATE ${PROJECT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
vf_set_project_options(vf_index_integration_tests)
gtest_discover_tests(vf_index_integration_tests
  DISCOVERY_MODE PRE_TEST
  PROPERTIES LABELS integration)

if(VF_BUILD_CLI)
  add_executable(vf_integration_tests
    integration/test_cli_pipeline.cpp
  )
  target_link_libraries(vf_integration_tests PRIVATE vf_cli_commands GTest::gtest_main)
  target_include_directories(vf_integration_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  vf_set_project_options(vf_integration_tests)
  gtest_discover_tests(vf_integration_tests
    DISCOVERY_MODE PRE_TEST
    PROPERTIES LABELS integration)

  # Smoke-test the real executable end to end.
  set(vf_cli_smoke_dir "${CMAKE_CURRENT_BINARY_DIR}/cli_smoke")
  add_test(NAME cli.version COMMAND vectorforge_cli --version)
  add_test(NAME cli.gen_base
           COMMAND vectorforge_cli gen-data --n 500 --dim 8 --seed 1 --out ${vf_cli_smoke_dir}/base.npy)
  add_test(NAME cli.gen_queries
           COMMAND vectorforge_cli gen-data --n 10 --dim 8 --seed 2 --format fvecs
                   --out ${vf_cli_smoke_dir}/queries.fvecs)
  add_test(NAME cli.ground_truth
           COMMAND vectorforge_cli ground-truth --base ${vf_cli_smoke_dir}/base.npy
                   --queries ${vf_cli_smoke_dir}/queries.fvecs --metric cosine --k 5
                   --out ${vf_cli_smoke_dir}/gt)
  add_test(NAME cli.rejects_bad_args COMMAND vectorforge_cli gen-data --n 0 --dim 8 --out x.npy)
  add_test(NAME cli.build
           COMMAND vectorforge_cli build --input ${vf_cli_smoke_dir}/base.npy --metric cosine
                   --M 8 --ef-construction 32 --out ${vf_cli_smoke_dir}/index.vfidx)
  add_test(NAME cli.info COMMAND vectorforge_cli info ${vf_cli_smoke_dir}/index.vfidx)
  add_test(NAME cli.verify COMMAND vectorforge_cli verify ${vf_cli_smoke_dir}/index.vfidx)
  add_test(NAME cli.search
           COMMAND vectorforge_cli search --index ${vf_cli_smoke_dir}/index.vfidx
                   --queries ${vf_cli_smoke_dir}/queries.fvecs --k 5 --ef 64 --gt ${vf_cli_smoke_dir}/gt)
  add_test(NAME cli.verify_rejects_non_index COMMAND vectorforge_cli verify ${vf_cli_smoke_dir}/base.npy)
  add_test(NAME cli.simd_invalid_request COMMAND vectorforge_cli verify ${vf_cli_smoke_dir}/index.vfidx)
  set_tests_properties(cli.simd_invalid_request PROPERTIES
                       ENVIRONMENT "VF_SIMD=fastest"
                       PASS_REGULAR_EXPRESSION "INVALID_ARGUMENT: VF_SIMD='fastest'"
                       FIXTURES_REQUIRED cli_index LABELS integration)
  set_tests_properties(cli.rejects_bad_args cli.verify_rejects_non_index PROPERTIES WILL_FAIL TRUE)
  set_tests_properties(cli.gen_base PROPERTIES FIXTURES_SETUP cli_base)
  set_tests_properties(cli.gen_queries PROPERTIES FIXTURES_SETUP cli_queries)
  set_tests_properties(cli.ground_truth PROPERTIES FIXTURES_REQUIRED "cli_base;cli_queries" FIXTURES_SETUP cli_gt)
  set_tests_properties(cli.build PROPERTIES FIXTURES_REQUIRED cli_base FIXTURES_SETUP cli_index)
  set_tests_properties(cli.info cli.verify PROPERTIES FIXTURES_REQUIRED cli_index)
  set_tests_properties(cli.search PROPERTIES FIXTURES_REQUIRED "cli_index;cli_queries;cli_gt")
  set_tests_properties(cli.verify_rejects_non_index PROPERTIES FIXTURES_REQUIRED cli_base)
  set_tests_properties(cli.version cli.gen_base cli.gen_queries cli.ground_truth cli.rejects_bad_args
                       cli.build cli.info cli.verify cli.search cli.verify_rejects_non_index
                       PROPERTIES LABELS integration)
  file(MAKE_DIRECTORY ${vf_cli_smoke_dir})
endif()
