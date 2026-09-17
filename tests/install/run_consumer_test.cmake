# Installs the current build into a temporary prefix, then configures, builds and runs
# tests/install/consumer against it with find_package(vectorforge).
#
#   cmake -DBUILD_DIR=... -DSOURCE_DIR=... -DWORK_DIR=... -DGENERATOR=... -DCXX_COMPILER=...
#         -DBUILD_TYPE=... [-DEXE_LINKER_FLAGS=...] -P run_consumer_test.cmake
#
# EXE_LINKER_FLAGS carries the parent build's flags (MinGW links its runtime statically).
foreach(var BUILD_DIR SOURCE_DIR WORK_DIR GENERATOR CXX_COMPILER BUILD_TYPE)
  if(NOT DEFINED ${var})
    message(FATAL_ERROR "${var} is required")
  endif()
endforeach()

set(prefix "${WORK_DIR}/prefix")
set(consumer_build "${WORK_DIR}/consumer")
file(REMOVE_RECURSE "${WORK_DIR}")

function(run)
  execute_process(COMMAND ${ARGV} RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "command failed (${rc}): ${ARGV}\n${out}\n${err}")
  endif()
endfunction()

run(${CMAKE_COMMAND} --install "${BUILD_DIR}" --prefix "${prefix}" --config "${BUILD_TYPE}")
foreach(expected include/vectorforge/collection.hpp include/vectorforge/version.hpp)
  if(NOT EXISTS "${prefix}/${expected}")
    message(FATAL_ERROR "missing installed file ${expected}")
  endif()
endforeach()

run(${CMAKE_COMMAND} -S "${SOURCE_DIR}/tests/install/consumer" -B "${consumer_build}"
    -G "${GENERATOR}" "-DCMAKE_CXX_COMPILER=${CXX_COMPILER}" "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
    "-DCMAKE_PREFIX_PATH=${prefix}" "-DCMAKE_EXE_LINKER_FLAGS=${EXE_LINKER_FLAGS}")
run(${CMAKE_COMMAND} --build "${consumer_build}" --config "${BUILD_TYPE}")

find_program(consumer_exe consumer PATHS "${consumer_build}" "${consumer_build}/${BUILD_TYPE}"
             NO_DEFAULT_PATH REQUIRED)
execute_process(COMMAND "${consumer_exe}" WORKING_DIRECTORY "${WORK_DIR}"
                RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
message(STATUS "${out}")
if(NOT rc EQUAL 0 OR NOT out MATCHES "nearest: 42")
  message(FATAL_ERROR "consumer failed (${rc}):\n${out}\n${err}")
endif()
file(REMOVE_RECURSE "${WORK_DIR}")
