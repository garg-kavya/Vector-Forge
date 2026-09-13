# Third-party dependencies, pinned by release tag AND archive SHA-256.
# Only tests and benchmarks use them; the core library has no third-party dependencies.

include(FetchContent)

set(VF_GTEST_VERSION 1.18.0)
set(VF_GTEST_SHA256 6e3191c1455468b3fc35a417fb565c1c5071aee1b7e7f85e30cf48a98d37d8b5)
set(VF_GBENCH_VERSION 1.9.5)
set(VF_GBENCH_SHA256 9631341c82bac4a288bef951f8b26b41f69021794184ece969f8473977eaa340)

if(VF_BUILD_TESTS)
  if(VF_USE_SYSTEM_DEPS)
    find_package(GTest REQUIRED)
  else()
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE) # match CMake's default /MD runtime on MSVC
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/refs/tags/v${VF_GTEST_VERSION}.tar.gz
      URL_HASH SHA256=${VF_GTEST_SHA256}
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
      SYSTEM)
    FetchContent_MakeAvailable(googletest)
  endif()
endif()

if(VF_BUILD_BENCHMARKS)
  if(VF_USE_SYSTEM_DEPS)
    find_package(benchmark REQUIRED)
  else()
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_LIBPFM OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
      benchmark
      URL https://github.com/google/benchmark/archive/refs/tags/v${VF_GBENCH_VERSION}.tar.gz
      URL_HASH SHA256=${VF_GBENCH_SHA256}
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
      SYSTEM)
    FetchContent_MakeAvailable(benchmark)
  endif()
endif()
