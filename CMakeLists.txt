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
option(VF_BUILD_CLI "Build the vectorforge CLI" ON)
option(VF_BUILD_SERVER "Build the HTTP server (vf_server, `vectorforge serve`)" ON)
option(VF_BUILD_PYTHON "Build the Python module in-tree (pip install ./python is the usual route)" OFF)
option(VF_ENABLE_AVX2 "Compile AVX2+FMA kernels for runtime dispatch (Phase 5)" ON)
option(VF_NATIVE "Compile everything for the host CPU (-march=native / /arch:AVX2); benchmark-only" OFF)
option(VF_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(VF_USE_SYSTEM_DEPS "Use find_package() for third-party deps instead of FetchContent" OFF)
option(VF_ENABLE_LTO "Enable interprocedural optimization for Release builds" OFF)
option(VF_BUILD_FUZZERS "Build libFuzzer targets (Clang only; combine with VF_SANITIZE)" OFF)
option(VF_ENABLE_ASSERTS "Keep VF_ASSERT checks in optimised builds" OFF)
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

if(VF_ENABLE_ASSERTS)
  add_compile_definitions(VF_ENABLE_ASSERTS=1)
endif()

if(VF_BUILD_FUZZERS)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "VF_BUILD_FUZZERS requires Clang (libFuzzer)")
  endif()
  # Coverage instrumentation for all code; only fuzz executables link the libFuzzer runtime.
  add_compile_options(-fsanitize=fuzzer-no-link)
endif()

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
    # Mark builds configured from a modified working tree (evaluated at configure time).
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" status --porcelain
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      OUTPUT_VARIABLE vf_git_status
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(vf_git_status)
      string(APPEND VF_GIT_SHA "-dirty")
    endif()
  endif()
endif()
set(VF_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/include")
configure_file(include/vectorforge/version.hpp.in "${VF_GENERATED_INCLUDE_DIR}/vectorforge/version.hpp" @ONLY)

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
if(VF_BUILD_PYTHON)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # the core is linked into a shared module
endif()

add_subdirectory(src)

if(VF_BUILD_TESTS OR VF_BUILD_BENCHMARKS OR VF_BUILD_CLI OR VF_BUILD_SERVER OR VF_BUILD_PYTHON)
  include(Dependencies)
endif()

if(VF_BUILD_SERVER)
  add_subdirectory(src/server)
endif()

if(VF_BUILD_CLI)
  add_subdirectory(apps/cli)
endif()

if(VF_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(VF_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(VF_BUILD_PYTHON)
  add_subdirectory(python)
endif()

# ---------------------------------------------------------------------------
# Installation: library, public headers, CLI, CMake package (find_package(vectorforge))
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
set(VF_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/vectorforge")
install(TARGETS vectorforge EXPORT vectorforgeTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY include/vectorforge DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.hpp")
install(FILES "${VF_GENERATED_INCLUDE_DIR}/vectorforge/version.hpp"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/vectorforge)
install(EXPORT vectorforgeTargets NAMESPACE vectorforge:: DESTINATION ${VF_INSTALL_CMAKEDIR})
configure_package_config_file(cmake/vectorforgeConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/vectorforgeConfig.cmake"
  INSTALL_DESTINATION ${VF_INSTALL_CMAKEDIR})
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/vectorforgeConfigVersion.cmake"
  COMPATIBILITY SameMinorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vectorforgeConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/vectorforgeConfigVersion.cmake"
        DESTINATION ${VF_INSTALL_CMAKEDIR})
install(FILES LICENSE README.md CHANGELOG.md DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/vectorforge)
if(VF_BUILD_CLI)
  install(TARGETS vectorforge_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

message(STATUS "VectorForge ${PROJECT_VERSION} (${VF_GIT_SHA}) build=${CMAKE_BUILD_TYPE} "
               "compiler=${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} sanitize='${VF_SANITIZE}'")
