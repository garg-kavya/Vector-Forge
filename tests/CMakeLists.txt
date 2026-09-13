include(GoogleTest)

# Unit tests ------------------------------------------------------------------------------------
add_executable(vf_unit_tests
  unit/test_aligned_alloc.cpp
  unit/test_checked_math.cpp
  unit/test_collection_flat.cpp
  unit/test_cpu_features.cpp
  unit/test_dataset_io.cpp
  unit/test_flat_backend.cpp
  unit/test_hnsw_determinism.cpp
  unit/test_hnsw_edge_cases.cpp
  unit/test_hnsw_graph.cpp
  unit/test_hnsw_tombstones.cpp
  unit/test_hnsw_validator.cpp
  unit/test_id_map.cpp
  unit/test_kernels.cpp
  unit/test_level_generator.cpp
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

# Allocation tests: separate binary because it replaces the global operator new/delete -----------
add_executable(vf_alloc_tests
  alloc/counting_allocator.cpp
  alloc/test_zero_alloc_search.cpp
)
target_link_libraries(vf_alloc_tests PRIVATE vectorforge::vectorforge GTest::gtest_main)
target_include_directories(vf_alloc_tests PRIVATE ${PROJECT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
vf_set_project_options(vf_alloc_tests)
gtest_discover_tests(vf_alloc_tests
  DISCOVERY_MODE PRE_TEST
  PROPERTIES LABELS unit)

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
  set_tests_properties(cli.rejects_bad_args PROPERTIES WILL_FAIL TRUE)
  set_tests_properties(cli.gen_base PROPERTIES FIXTURES_SETUP cli_base)
  set_tests_properties(cli.gen_queries PROPERTIES FIXTURES_SETUP cli_queries)
  set_tests_properties(cli.ground_truth PROPERTIES FIXTURES_REQUIRED "cli_base;cli_queries")
  set_tests_properties(cli.version cli.gen_base cli.gen_queries cli.ground_truth cli.rejects_bad_args
                       PROPERTIES LABELS integration)
  file(MAKE_DIRECTORY ${vf_cli_smoke_dir})
endif()
