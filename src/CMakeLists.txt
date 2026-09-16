set(VF_CORE_SOURCES
  collection/collection.cpp
  collection/index_reader.cpp
  collection/index_writer.cpp
  collection/snapshot.cpp
  concurrency/thread_pool.cpp
  index/flat_backend.cpp
  index/hnsw/hnsw_backend.cpp
  index/hnsw/hnsw_graph.cpp
  index/hnsw/hnsw_insert.cpp
  index/hnsw/hnsw_io.cpp
  index/hnsw/hnsw_search.cpp
  index/hnsw/hnsw_validator.cpp
  storage/atomic_file.cpp
  storage/binary_io.cpp
  storage/crc32c.cpp
  storage/format.cpp
  storage/id_map.cpp
  util/dataset_io.cpp
  util/recall.cpp
  util/synthetic.cpp
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

if(WIN32)
  list(APPEND VF_CORE_SOURCES storage/atomic_file_win32.cpp storage/mapped_file_win32.cpp)
else()
  list(APPEND VF_CORE_SOURCES storage/atomic_file_posix.cpp storage/mapped_file_posix.cpp)
endif()

add_library(vectorforge STATIC ${VF_CORE_SOURCES})
add_library(vectorforge::vectorforge ALIAS vectorforge)

target_include_directories(vectorforge
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${VF_GENERATED_INCLUDE_DIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})

if(WIN32)
  # Windows 10 API level (PrefetchVirtualMemory needs Windows 8 or later).
  target_compile_definitions(vectorforge PRIVATE _WIN32_WINNT=0x0A00 WINVER=0x0A00)
endif()

vf_set_project_options(vectorforge)

find_package(Threads REQUIRED)
target_link_libraries(vectorforge PUBLIC Threads::Threads)

# AVX2 + FMA kernels: the only library translation unit with non-baseline instruction-set flags,
# built as an object library so the flags cannot reach other sources (docs/simd.md).
vf_avx2_kernels_enabled(VF_AVX2_KERNELS)
set(VF_AVX2_KERNELS ${VF_AVX2_KERNELS} CACHE INTERNAL "AVX2 kernels are compiled into vectorforge")
if(VF_AVX2_KERNELS)
  add_library(vf_simd_avx2 OBJECT simd/kernels_avx2.cpp)
  target_include_directories(vf_simd_avx2 PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
  vf_set_project_options(vf_simd_avx2)
  vf_avx2_compile_options(vf_avx2_flags)
  target_compile_options(vf_simd_avx2 PRIVATE ${vf_avx2_flags})
  target_sources(vectorforge PRIVATE $<TARGET_OBJECTS:vf_simd_avx2>)
  target_compile_definitions(vectorforge PRIVATE VF_HAVE_AVX2_KERNELS=1)
endif()

# Floating-point semantics per kernel translation unit (see cmake/SimdFlags.cmake).
vf_set_strict_fp_source(simd/kernels_scalar.cpp)
vf_set_autovec_source(simd/kernels_scalar_autovec.cpp)
# Distance formulas outside the kernels must not be contracted either: HNSW graphs are compared
# byte for byte across builds (docs/DESIGN.md §15.3) and must match FlatBackend distances.
vf_set_strict_fp_source(index/flat_backend.cpp)
vf_set_strict_fp_source(index/hnsw/hnsw_insert.cpp)
vf_set_strict_fp_source(index/hnsw/hnsw_search.cpp)
