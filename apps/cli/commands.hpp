#pragma once

// `vectorforge` CLI commands, callable in-process (the executable only parses arguments).

#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <string>

#include <vectorforge/status.hpp>
#include <vectorforge/types.hpp>

#include "util/synthetic.hpp"

namespace vf::cli {

enum class DatasetFormat : std::uint8_t { Npy, Fvecs };

struct GenDataOptions {
  std::uint64_t rows = 0;
  vf::detail::SyntheticSpec spec{};
  DatasetFormat format = DatasetFormat::Npy;
  std::filesystem::path out;
};

// Streams `rows` synthetic float32 vectors to `out` (O(dim) memory).
[[nodiscard]] Status run_gen_data(const GenDataOptions& options, std::ostream& log);

struct GroundTruthOptions {
  std::filesystem::path base;     // .npy or .fvecs, float32; row i gets id i
  std::filesystem::path queries;  // .npy or .fvecs, float32
  Metric metric = Metric::L2;
  std::uint32_t k = 100;
  // Writes <out_prefix>.ids.npy (int64, nq x k, -1 padding) and
  // <out_prefix>.distances.npy (float32, nq x k, +inf padding).
  std::filesystem::path out_prefix;
};

// Exact k-NN ground truth with a Flat collection (single-threaded in this version).
[[nodiscard]] Status run_ground_truth(const GroundTruthOptions& options, std::ostream& log);

// Output paths used by run_ground_truth.
[[nodiscard]] std::filesystem::path ground_truth_ids_path(const std::filesystem::path& prefix);
[[nodiscard]] std::filesystem::path ground_truth_distances_path(
    const std::filesystem::path& prefix);

}  // namespace vf::cli
