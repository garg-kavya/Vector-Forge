#pragma once

// Deterministic test data helpers (platform-independent; see core/rng.hpp).

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <system_error>
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

// Small integer-valued components in [lo, hi]. With |x| <= 8 and dim <= 64, every squared L2
// distance and dot product is an integer below 2^24, so float32 arithmetic is exact and results can
// be compared for exact equality against a double-precision reference - including exact ties.
inline std::vector<float> integer_vector(vf::detail::Xoshiro256ss& rng, std::size_t dim,
                                         int lo = -8, int hi = 8) {
  std::vector<float> v(dim);
  const auto span = static_cast<std::uint64_t>(hi - lo + 1);
  for (float& x : v) {
    x = static_cast<float>(lo + static_cast<int>(vf::detail::uniform_below(rng, span)));
  }
  return v;
}

inline std::vector<float> integer_matrix(vf::detail::Xoshiro256ss& rng, std::size_t rows,
                                         std::size_t dim, int lo = -8, int hi = 8) {
  return integer_vector(rng, rows * dim, lo, hi);
}

// Clustered data that is bit-identical on every platform (no std::log, unlike the Gaussian
// mixture generator): `clusters` centres uniform in [-1, 1)^dim, each row = a uniformly chosen
// centre plus independent uniform noise in [-spread, spread) per component.
class ClusteredData {
 public:
  ClusteredData(std::uint64_t mixture_seed, std::size_t dim, std::size_t clusters, float spread)
      : dim_(dim), clusters_(clusters), spread_(spread) {
    vf::detail::Xoshiro256ss rng(mixture_seed);
    centres_ = random_matrix(rng, clusters, dim);
  }

  [[nodiscard]] std::vector<float> rows(std::uint64_t seed, std::size_t count) const {
    vf::detail::Xoshiro256ss rng(seed);
    std::vector<float> out(count * dim_);
    for (std::size_t r = 0; r < count; ++r) {
      const auto c = static_cast<std::size_t>(vf::detail::uniform_below(rng, clusters_));
      for (std::size_t j = 0; j < dim_; ++j) {
        out[(r * dim_) + j] =
            centres_[(c * dim_) + j] + vf::detail::uniform_float(rng(), -spread_, spread_);
      }
    }
    return out;
  }

 private:
  std::size_t dim_;
  std::size_t clusters_;
  float spread_;
  std::vector<float> centres_;
};

// Unique temporary directory removed on destruction.
class ScopedTempDir {
 public:
  explicit ScopedTempDir(const std::string& tag) {
    vf::detail::Xoshiro256ss rng(static_cast<std::uint64_t>(
        std::filesystem::file_time_type::clock::now().time_since_epoch().count()));
    path_ = std::filesystem::temp_directory_path() /
            ("vectorforge_" + tag + "_" + std::to_string(rng() & 0xFFFFFFFFULL));
    std::filesystem::create_directories(path_);
  }
  ScopedTempDir(const ScopedTempDir&) = delete;
  ScopedTempDir& operator=(const ScopedTempDir&) = delete;
  ScopedTempDir(ScopedTempDir&&) = delete;
  ScopedTempDir& operator=(ScopedTempDir&&) = delete;
  ~ScopedTempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }

  [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }
  [[nodiscard]] std::filesystem::path file(const std::string& name) const { return path_ / name; }

 private:
  std::filesystem::path path_;
};

}  // namespace vf::test
