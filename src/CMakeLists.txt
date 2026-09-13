set(VF_CORE_SOURCES
  core/build_info.cpp
  core/config.cpp
  core/status.cpp
  core/types.cpp
  core/validation.cpp
  core/vector_ops.cpp
  simd/cpu_features.cpp
  simd/dispatch.cpp
  simd/kernels_scalar.cpp
  simd/kernels_scalar_autovec.cpp
  storage/vector_store.cpp
)

add_library(vectorforge STATIC ${VF_CORE_SOURCES})
add_library(vectorforge::vectorforge ALIAS vectorforge)

target_include_directories(vectorforge
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${VF_GENERATED_INCLUDE_DIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})

vf_set_project_options(vectorforge)

# Floating-point semantics per kernel translation unit (see cmake/SimdFlags.cmake).
vf_set_strict_fp_source(simd/kernels_scalar.cpp)
vf_set_autovec_source(simd/kernels_scalar_autovec.cpp)
