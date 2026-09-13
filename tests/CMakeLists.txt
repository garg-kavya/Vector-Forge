include(GoogleTest)

add_executable(vf_unit_tests
  unit/test_aligned_alloc.cpp
  unit/test_checked_math.cpp
  unit/test_cpu_features.cpp
  unit/test_kernels.cpp
  unit/test_normalize.cpp
  unit/test_rng.cpp
  unit/test_smoke.cpp
  unit/test_status.cpp
  unit/test_types_config.cpp
  unit/test_validation.cpp
  unit/test_vector_store.cpp
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
