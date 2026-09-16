#pragma once

// Request limits of the HTTP server (docs/DESIGN.md §13.5, docs/http-api.md "Limits").

#include <cstddef>
#include <cstdint>

namespace vf::server {

struct Limits {
  std::size_t max_body_bytes = std::size_t{64} << 20U;  // larger bodies: 413
  std::size_t max_batch = 10000;                        // vectors per insert or search batch
  std::uint32_t max_k = 1000;
  std::uint32_t max_ef = 4096;
  std::uint32_t max_dim = 65536;  // for collection creation
};

}  // namespace vf::server
