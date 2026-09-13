#pragma once

// Deterministic test data helpers (platform-independent; see core/rng.hpp).

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/rng.hpp"

namespace vf::test {

inline std::vector<float> random_vector(vf::detail::Xoshiro256ss& rng, std::size_t dim,
                                        float lo = -1.0F, float hi = 1.0F) {
  std::vector<float> v(dim);
  for (float& x : v) {
    x = vf::detail::uniform_float(rng(), lo, hi);
  }
  return v;
}

inline std::vector<float> random_matrix(vf::detail::Xoshiro256ss& rng, std::size_t rows,
                                        std::size_t dim, float lo = -1.0F, float hi = 1.0F) {
  return random_vector(rng, rows * dim, lo, hi);
}

}  // namespace vf::test
