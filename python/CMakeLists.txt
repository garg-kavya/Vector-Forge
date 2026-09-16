# Python extension module `vectorforge._vectorforge` (docs/python-api.md).
#
# Two ways in:
#   - `pip install ./python` (scikit-build-core): this file is the top-level project and builds the
#     C++ core from the repository root; pybind11 comes from the build requirements.
#   - `-DVF_BUILD_PYTHON=ON` on the root project: the core already exists and pybind11 is fetched
#     by cmake/Dependencies.cmake; the module is placed next to the package sources for testing.
cmake_minimum_required(VERSION 3.25)

if(NOT TARGET vectorforge::vectorforge)
  project(vectorforge_python VERSION 0.1.0 LANGUAGES CXX)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  set(VF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(VF_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
  set(VF_BUILD_CLI OFF CACHE BOOL "" FORCE)
  set(VF_BUILD_SERVER OFF CACHE BOOL "" FORCE)
  set(VF_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/.. vectorforge-core EXCLUDE_FROM_ALL)
  find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  set(vf_python_standalone ON)
endif()

pybind11_add_module(_vectorforge MODULE src/vectorforge/_bindings.cpp)
target_link_libraries(_vectorforge PRIVATE vectorforge::vectorforge)
target_compile_features(_vectorforge PRIVATE cxx_std_20)
if(MSVC)
  target_compile_options(_vectorforge PRIVATE /W4 /utf-8 /permissive-)
else()
  target_compile_options(_vectorforge PRIVATE -Wall -Wextra)
endif()

if(vf_python_standalone)
  install(TARGETS _vectorforge LIBRARY DESTINATION vectorforge)
else()
  # In-tree build: put the module into the source package so `PYTHONPATH=python/src` imports it.
  set_target_properties(_vectorforge PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/package/vectorforge"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_BINARY_DIR}/package/vectorforge"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_BINARY_DIR}/package/vectorforge")
  add_custom_command(TARGET _vectorforge POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/src/vectorforge/__init__.py
            ${CMAKE_CURRENT_SOURCE_DIR}/src/vectorforge/py.typed
            ${CMAKE_CURRENT_BINARY_DIR}/package/vectorforge/)
  if(VF_BUILD_TESTS)
    add_test(NAME python.pytest
             COMMAND ${Python_EXECUTABLE} -m pytest -q ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    set_tests_properties(python.pytest PROPERTIES
      LABELS python
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/package")
  endif()
endif()
