# Commands live in a library so tests can run them in-process; the executable only parses arguments.
add_library(vf_cli_commands STATIC
  cmd_build.cpp
  cmd_gen.cpp
  cmd_gt.cpp
  cmd_info.cpp
  cmd_search.cpp
  pool.cpp
)
target_include_directories(vf_cli_commands
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
  PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(vf_cli_commands PUBLIC vectorforge::vectorforge)
if(VF_BUILD_SERVER)
  target_sources(vf_cli_commands PRIVATE cmd_serve.cpp)
  target_link_libraries(vf_cli_commands PUBLIC vectorforge::server)
  target_compile_definitions(vf_cli_commands PUBLIC VF_HAVE_SERVER=1)
endif()
vf_set_project_options(vf_cli_commands)

add_executable(vectorforge_cli main.cpp)
set_target_properties(vectorforge_cli PROPERTIES OUTPUT_NAME vectorforge)
target_link_libraries(vectorforge_cli PRIVATE vf_cli_commands CLI11::CLI11)
vf_set_project_options(vectorforge_cli)
