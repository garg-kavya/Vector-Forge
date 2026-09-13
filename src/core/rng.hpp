#pragma once

// Deterministic, platform-independent pseudo-random number generation.
//
// The standard <random> distributions (uniform_real_distribution, normal_distribution, ...) have
// implementation-defined output, so identical seeds produce different values on MSVC, libstdc++
// and libc++. VectorForge needs bit-identical behaviour everywhere (HNSW level assignment,
// synthetic datasets, tests), so it uses these explicitly specified generators and conversions.
//
// SplitMix64 and xoshiro256** follow the public-domain reference algorithms by Sebastiano Vigna
// (https://prng.di.unimi.it/). Neither is cryptographically secure.

#include <array>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>

namespace vf::detail {

inline constexpr std::uint64_t kSplitMix64Increment = 0x9E3779B97F4A7C15ULL;

// Stateless SplitMix64 output function applied to `state + increment`; a high-quality 64-bit mixer.
// splitmix64(x) equals the first output of SplitMix64 seeded with x.
[[nodiscard]] constexpr std::uint64_t splitmix64(std::uint64_t x) noexcept {
  std::uint64_t z = x + kSplitMix64Increment;
  z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31U);
}

class SplitMix64 {
 public:
  using result_type = std::uint64_t;

  constexpr explicit SplitMix64(std::uint64_t seed) noexcept : state_(seed) {}

  constexpr result_type operator()() noexcept {
    const std::uint64_t out = splitmix64(state_);
    state_ += kSplitMix64Increment;
    return out;
  }

  static constexpr result_type min() noexcept { return 0; }
  static constexpr result_type max() noexcept { return std::numeric_limits<result_type>::max(); }

 private:
  std::uint64_t state_;
};

// xoshiro256**: fast all-purpose generator with 2^256 - 1 period. Satisfies
// std::uniform_random_bit_generator.
class Xoshiro256ss {
 public:
  using result_type = std::uint64_t;

  // Seeds the 256-bit state from SplitMix64(seed), as recommended by the reference implementation;
  // this can never produce the forbidden all-zero state in practice.
  constexpr explicit Xoshiro256ss(std::uint64_t seed) noexcept {
    SplitMix64 seeder(seed);
    for (std::uint64_t& word : state_) {
      word = seeder();
    }
  }

  constexpr result_type operator()() noexcept {
    const std::uint64_t result = std::rotl(state_[1] * 5U, 7) * 9U;
    const std::uint64_t t = state_[1] << 17U;
    state_[2] ^= state_[0];
    state_[3] ^= state_[1];
    state_[1] ^= state_[2];
    state_[0] ^= state_[3];
    state_[2] ^= t;
    state_[3] = std::rotl(state_[3], 45);
    return result;
  }

  static constexpr result_type min() noexcept { return 0; }
  static constexpr result_type max() noexcept { return std::numeric_limits<result_type>::max(); }

 private:
  std::array<std::uint64_t, 4> state_{};
};

// Maps 64 random bits to a double in [0, 1) using the top 53 bits (exactly representable grid).
[[nodiscard]] constexpr double to_unit_interval(std::uint64_t bits) noexcept {
  return static_cast<double>(bits >> 11U) * 0x1.0p-53;
}

// Maps 64 random bits to a double in (0, 1]. Never returns 0, so std::log() of it is finite.
[[nodiscard]] constexpr double to_unit_interval_open_zero(std::uint64_t bits) noexcept {
  return static_cast<double>((bits >> 11U) + 1U) * 0x1.0p-53;
}

// High 64 bits of the 128-bit product a * b (portable; MSVC has no unsigned __int128).
[[nodiscard]] constexpr std::uint64_t mul_high_u64(std::uint64_t a, std::uint64_t b) noexcept {
  const std::uint64_t a_lo = a & 0xFFFFFFFFULL;
  const std::uint64_t a_hi = a >> 32U;
  const std::uint64_t b_lo = b & 0xFFFFFFFFULL;
  const std::uint64_t b_hi = b >> 32U;
  const std::uint64_t lo_lo = a_lo * b_lo;
  const std::uint64_t hi_lo = a_hi * b_lo;
  const std::uint64_t lo_hi = a_lo * b_hi;
  const std::uint64_t hi_hi = a_hi * b_hi;
  const std::uint64_t cross = (lo_lo >> 32U) + (hi_lo & 0xFFFFFFFFULL) + lo_hi;
  return hi_hi + (hi_lo >> 32U) + (cross >> 32U);
}

// Unbiased integer in [0, bound) using Lemire's multiply-and-reject method. Precondition: bound >
// 0.
template <class Generator>
  requires std::same_as<typename Generator::result_type, std::uint64_t>
[[nodiscard]] constexpr std::uint64_t uniform_below(Generator& gen, std::uint64_t bound) noexcept {
  std::uint64_t x = gen();
  std::uint64_t low = x * bound;  // low 64 bits of the product
  if (low < bound) {
    const std::uint64_t threshold = (std::uint64_t{0} - bound) % bound;  // 2^64 mod bound
    while (low < threshold) {
      x = gen();
      low = x * bound;
    }
  }
  return mul_high_u64(x, bound);
}

// Uniform float in [lo, hi) computed from 64 random bits in a fully specified way.
[[nodiscard]] inline float uniform_float(std::uint64_t bits, float lo, float hi) noexcept {
  const double u = to_unit_interval(bits);
  const auto lo_d = static_cast<double>(lo);
  const auto hi_d = static_cast<double>(hi);
  const double value = lo_d + (hi_d - lo_d) * u;
  const auto result = static_cast<float>(value);
  // Rounding to float can land exactly on `hi`; keep the half-open contract.
  return result < hi ? result : std::nextafter(hi, lo);
}

}  // namespace vf::detail
