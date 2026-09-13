#pragma once

// Seeded synthetic datasets for tests and benchmarks (docs/DESIGN.md §16.2).
//
//   Uniform         - components i.i.d. uniform in [-1, 1). Bit-identical on every platform.
//   GaussianMixture - `clusters` centres uniform in [-center_range, center_range)^dim (drawn from
//                     `mixture_seed`), each row = centre[c] + spread * N(0, I) with c uniform.
//                     Normal deviates use the Marsaglia polar method, which calls std::log and
//                     std::sqrt; sqrt is correctly rounded by IEEE 754 but log is not, so results
//                     are reproducible per platform/C library but not guaranteed bit-identical
//                     across them.
//
// Base and query sets drawn from the same mixture share `mixture_seed` and differ in `seed`.
// Rows are produced sequentially, so arbitrarily large datasets can be streamed to disk.

#include <cstdint>
#include <span>
#include <vector>

#include <vectorforge/status.hpp>

#include "core/rng.hpp"

namespace vf::detail {

enum class SyntheticDistribution : std::uint8_t { Uniform, GaussianMixture };

struct SyntheticSpec {
  static constexpr std::uint32_t kMaxClusters = 1U << 20U;

  SyntheticDistribution distribution = SyntheticDistribution::GaussianMixture;
  std::uint32_t dim = 0;
  std::uint32_t clusters = 100;
  float center_range = 1.0F;
  float spread = 0.1F;
  std::uint64_t seed = 1;
  std::uint64_t mixture_seed = 42;

  [[nodiscard]] Status validate() const;
};

class SyntheticGenerator {
 public:
  // Errors: InvalidArgument (invalid spec).
  [[nodiscard]] static Result<SyntheticGenerator> create(const SyntheticSpec& spec);

  // Writes the next row into `row` (size must equal spec.dim).
  void next_row(std::span<float> row) noexcept;

  [[nodiscard]] const SyntheticSpec& spec() const noexcept { return spec_; }

 private:
  explicit SyntheticGenerator(const SyntheticSpec& spec);
  double next_normal() noexcept;

  SyntheticSpec spec_;
  Xoshiro256ss rng_;
  std::vector<float> centers_;  // clusters * dim
  double spare_normal_ = 0.0;
  bool has_spare_ = false;
};

}  // namespace vf::detail
