# HTTP server library (docs/http-api.md). Depends on the public library API only.
add_library(vf_server STATIC
  json_codec.cpp
  server.cpp
)
add_library(vectorforge::server ALIAS vf_server)
target_include_directories(vf_server PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(vf_server
  PUBLIC vectorforge::vectorforge vf_httplib nlohmann_json::nlohmann_json)
vf_set_project_options(vf_server)
