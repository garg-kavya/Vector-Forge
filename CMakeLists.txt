cmake_minimum_required(VERSION 3.25)

project(
  VectorForge
  VERSION 0.1.0
  DESCRIPTION "High-performance vector similarity search engine (HNSW, SIMD, C++20)"
  LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(VF_BUILD_TESTS "Build unit/integration tests (GoogleTest)" ON)
option(VF_BUILD_BENCHMARKS "Build micro benchmarks (Google Benchmark)" ON)
option(VF_BUILD_CLI "Build the vectorforge CLI (Phase 2+)" OFF)
option(VF_BUILD_SERVER "Build the HTTP server library (Phase 7)" OFF)
option(VF_BUILD_PYTHON "Build Python bindings (Phase 8)" OFF)
option(VF_ENABLE_AVX2 "Compile AVX2+FMA kernels for runtime dispatch (Phase 5)" ON)
option(VF_NATIVE "Compile everything for the host CPU (-march=native / /arch:AVX2); benchmark-only" OFF)
option(VF_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(VF_USE_SYSTEM_DEPS "Use find_package() for third-party deps instead of FetchContent" OFF)
option(VF_ENABLE_LTO "Enable interprocedural optimization for Release builds" OFF)
set(VF_SANITIZE "" CACHE STRING "Sanitizers: '', 'address', 'address;undefined', 'thread'")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Language / toolchain settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# /Z7 instead of /Zi: no shared PDB server contention under Ninja.
set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<$<CONFIG:Debug,RelWithDebInfo>:Embedded>")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerWarnings)
include(Sanitizers)
include(SimdFlags)

vf_configure_sanitizers() # global: must precede dependencies so they are instrumented too

if(VF_NATIVE)
  if(MSVC)
    add_compile_options(/arch:AVX2)
  else()
    add_compile_options(-march=native)
  endif()
endif()

if(VF_ENABLE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT vf_ipo_ok OUTPUT vf_ipo_msg)
  if(vf_ipo_ok)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
  else()
    message(WARNING "LTO requested but not supported: ${vf_ipo_msg}")
  endif()
endif()

# ---------------------------------------------------------------------------
# Version header
# ---------------------------------------------------------------------------
find_package(Git QUIET)
set(VF_GIT_SHA "unknown")
if(GIT_FOUND)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse --short=12 HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE vf_git_out
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE vf_git_rc)
  if(vf_git_rc EQUAL 0 AND vf_git_out)
    set(VF_GIT_SHA "${vf_git_out}")
  endif()
endif()
set(VF_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/include")
configure_file(include/vectorforge/version.hpp.in "${VF_GENERATED_INCLUDE_DIR}/vectorforge/version.hpp" @ONLY)

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
add_subdirectory(src)

if(VF_BUILD_TESTS OR VF_BUILD_BENCHMARKS)
  include(Dependencies)
endif()

if(VF_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(VF_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

foreach(vf_future IN ITEMS VF_BUILD_CLI VF_BUILD_SERVER VF_BUILD_PYTHON)
  if(${vf_future})
    message(STATUS "${vf_future}=ON has no effect yet (component not implemented in this phase)")
  endif()
endforeach()

message(STATUS "VectorForge ${PROJECT_VERSION} (${VF_GIT_SHA}) build=${CMAKE_BUILD_TYPE} "
               "compiler=${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} sanitize='${VF_SANITIZE}'")
