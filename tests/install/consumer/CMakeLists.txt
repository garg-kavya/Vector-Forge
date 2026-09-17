# A project that uses an installed VectorForge (tests/install/run_consumer_test.cmake builds it).
cmake_minimum_required(VERSION 3.25)
project(vectorforge_consumer LANGUAGES CXX)

find_package(vectorforge 0.1 REQUIRED CONFIG)

add_executable(consumer ../../../examples/cpp/quickstart.cpp)
target_link_libraries(consumer PRIVATE vectorforge::vectorforge)
target_compile_features(consumer PRIVATE cxx_std_20)
