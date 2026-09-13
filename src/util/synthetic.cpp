#include "util/synthetic.hpp"

#include <cmath>
#include <cstddef>
#include <string>

#include <vectorforge/types.hpp>

#include "core/assert.hpp"
#include "core/checked_math.hpp"

namespace vf::detail {

Status SyntheticSpec::validate() const {
  if (dim < 1 || dim > kMaxDim) {
    return Status::invalid_argument("dim must be in [1, " + std::to_string(kMaxDim) + "]");
  }
  if (distribution == SyntheticDistribution::GaussianMixture) {
    if (clusters < 1 || clusters > kMaxClusters) {
      return Status::invalid_argument("clusters must be in [1, " + std::to_string(kMaxClusters) +
                                      "]");
    }
    const auto centre_values = checked_mul(std::uint64_t{clusters}, std::uint64_t{dim});
    if (!centre_values || *centre_values > (std::uint64_t{1} << 28U)) {
      return Status::invalid_argument("clusters * dim exceeds 2^28 centre components");
    }
    // Keep generated components well inside the accepted magnitude.
    const auto out_of_range = [](float x) { return !std::isfinite(x) || x < 0.0F || x > 1e6F; };
    if (out_of_range(center_range) || out_of_range(spread)) {
      return Status::invalid_argument("center_range and spread must be in [0, 1e6]");
    }
  } else if (distribution != SyntheticDistribution::Uniform) {
    return Status::invalid_argument("unknown synthetic distribution");
  }
  return {};
}

Result<SyntheticGenerator> SyntheticGenerator::create(const SyntheticSpec& spec) {
  VF_RETURN_IF_ERROR(spec.validate());
  return SyntheticGenerator(spec);
}

SyntheticGenerator::SyntheticGenerator(const SyntheticSpec& spec) : spec_(spec), rng_(spec.seed) {
  if (spec_.distribution == SyntheticDistribution::GaussianMixture) {
    Xoshiro256ss centre_rng(spec_.mixture_seed);
    centers_.resize(static_cast<std::size_t>(spec_.clusters) * spec_.dim);
    for (float& c : centers_) {
      c = uniform_float(centre_rng(), -spec_.center_range, spec_.center_range);
    }
  }
}

double SyntheticGenerator::next_normal() noexcept {
  if (has_spare_) {
    has_spare_ = false;
    return spare_normal_;
  }
  double u = 0.0;
  double v = 0.0;
  double s = 0.0;
  do {
    u = (2.0 * to_unit_interval(rng_())) - 1.0;
    v = (2.0 * to_unit_interval(rng_())) - 1.0;
    s = (u * u) + (v * v);
  } while (s >= 1.0 || s == 0.0);
  const double scale = std::sqrt(-2.0 * std::log(s) / s);
  spare_normal_ = v * scale;
  has_spare_ = true;
  return u * scale;
}

void SyntheticGenerator::next_row(std::span<float> row) noexcept {
  VF_ASSERT(row.size() == spec_.dim, "SyntheticGenerator::next_row: wrong row size");
  if (spec_.distribution == SyntheticDistribution::Uniform) {
    for (float& x : row) {
      x = uniform_float(rng_(), -1.0F, 1.0F);
    }
    return;
  }
  const auto cluster = static_cast<std::size_t>(uniform_below(rng_, spec_.clusters));
  const float* centre = centers_.data() + (cluster * spec_.dim);
  const auto spread = static_cast<double>(spec_.spread);
  for (std::size_t j = 0; j < row.size(); ++j) {
    row[j] = static_cast<float>(static_cast<double>(centre[j]) + (spread * next_normal()));
  }
}

}  // namespace vf::detail
