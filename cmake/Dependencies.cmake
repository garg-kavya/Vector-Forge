# Third-party dependencies, pinned by release tag AND archive SHA-256.
# The core library has no third-party dependencies; the CLI, the server, tests and benchmarks
# use the ones below.

include(FetchContent)

set(VF_GTEST_VERSION 1.18.0)
set(VF_GTEST_SHA256 6e3191c1455468b3fc35a417fb565c1c5071aee1b7e7f85e30cf48a98d37d8b5)
set(VF_GBENCH_VERSION 1.9.5)
set(VF_GBENCH_SHA256 9631341c82bac4a288bef951f8b26b41f69021794184ece969f8473977eaa340)
set(VF_CLI11_VERSION 2.7.2)
set(VF_CLI11_SHA256 46eef3101da70852ec7af026e09d485ccee81813331c8c6052d39344443b83da)
set(VF_HTTPLIB_VERSION 0.54.1)
set(VF_HTTPLIB_SHA256 7310f5312e1423830d649b38ed028e9db86303a979ccbfdbd1c4b1574f422dfb)
set(VF_JSON_VERSION 3.12.0)
set(VF_JSON_SHA256 42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa)

if(VF_BUILD_CLI)
  if(VF_USE_SYSTEM_DEPS)
    find_package(CLI11 REQUIRED)
  else()
    set(CLI11_PRECOMPILED OFF CACHE BOOL "" FORCE)
    set(CLI11_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(CLI11_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(CLI11_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(CLI11_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
      cli11
      URL https://github.com/CLIUtils/CLI11/archive/refs/tags/v${VF_CLI11_VERSION}.tar.gz
      URL_HASH SHA256=${VF_CLI11_SHA256}
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
      SYSTEM)
    FetchContent_MakeAvailable(cli11)
  endif()
endif()

if(VF_BUILD_SERVER)
  # vf_httplib: cpp-httplib as a header-only target without TLS or compression (its own CMake
  # project probes for OpenSSL/zlib/brotli, which the server does not use).
  add_library(vf_httplib INTERFACE)
  if(VF_USE_SYSTEM_DEPS)
    find_package(httplib REQUIRED)
    find_package(nlohmann_json REQUIRED)
    target_link_libraries(vf_httplib INTERFACE httplib::httplib)
  else()
    FetchContent_Declare(
      cpp_httplib
      URL https://github.com/yhirose/cpp-httplib/archive/refs/tags/v${VF_HTTPLIB_VERSION}.tar.gz
      URL_HASH SHA256=${VF_HTTPLIB_SHA256}
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
      SOURCE_SUBDIR vf-header-only  # no CMakeLists.txt there: only download
      SYSTEM)
    FetchContent_MakeAvailable(cpp_httplib)
    target_include_directories(vf_httplib SYSTEM INTERFACE ${cpp_httplib_SOURCE_DIR})

    set(JSON_BuildTests OFF CACHE INTERNAL "")
    set(JSON_Install OFF CACHE INTERNAL "")
    FetchContent_Declare(
      nlohmann_json
      URL https://github.com/nlohmann/json/releases/download/v${VF_JSON_VERSION}/json.tar.xz
      URL_HASH SHA256=${VF_JSON_SHA256}
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
      SYSTEM)
    FetchContent_MakeAvailable(nlohmann_json)
  endif()
  find_package(Threads REQUIRED)
  target_link_libraries(vf_httplib INTERFACE Threads::Threads)
  if(WIN32)
    # cpp-httplib requires Windows 10 APIs and Winsock.
    target_compile_definitions(vf_httplib INTERFACE _WIN32_WINNT=0x0A00 WIN32_LEAN_AND_MEAN NOMINMAX)
    target_link_libraries(vf_httplib INTERFACE ws2_32)
  endif()
endif()

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
