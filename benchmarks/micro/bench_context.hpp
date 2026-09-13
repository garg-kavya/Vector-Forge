#pragma once

namespace vf::bench {

// Adds environment metadata (CPU features, SIMD tier, ...) to the benchmark JSON context.
void add_extra_context();

}  // namespace vf::bench
