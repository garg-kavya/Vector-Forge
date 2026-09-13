add_library(vectorforge STATIC
  core/build_info.cpp
)
add_library(vectorforge::vectorforge ALIAS vectorforge)

target_include_directories(vectorforge
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${VF_GENERATED_INCLUDE_DIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})

vf_set_project_options(vectorforge)
