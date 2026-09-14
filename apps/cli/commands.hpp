#pragma once

// `vectorforge` CLI commands, callable in-process (the executable only parses arguments).

#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <optional>
#include <string>

#include <vectorforge/config.hpp>
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

// Output paths used by run_ground_truth and run_search.
[[nodiscard]] std::filesystem::path ground_truth_ids_path(const std::filesystem::path& prefix);
[[nodiscard]] std::filesystem::path ground_truth_distances_path(
    const std::filesystem::path& prefix);

struct BuildOptions {
  std::filesystem::path input;  // .npy or .fvecs, float32
  // Optional external ids: .npy with int64 or uint64 values, one per input row (default: row
  // numbers).
  std::filesystem::path ids;
  CollectionConfig config;  // dim is taken from the input
  std::filesystem::path out;
};

// Builds a collection from a dataset file and saves it (single-threaded in this version).
[[nodiscard]] Status run_build(const BuildOptions& options, std::ostream& log);

struct SearchOptions {
  std::filesystem::path index;
  std::filesystem::path queries;  // .npy or .fvecs, float32
  std::uint32_t k = 10;
  std::optional<std::uint32_t> ef_search;
  bool use_mmap = true;
  // Optional: writes <out_prefix>.ids.npy (int64, -1 padding) and <out_prefix>.distances.npy.
  std::filesystem::path out_prefix;
  // Optional ground truth: a prefix written by ground-truth (tie-tolerant recall), or a TEXMEX
  // .ivecs file of neighbour ids such as SIFT's groundtruth (id-overlap recall).
  std::filesystem::path ground_truth;
};

// Searches every query one at a time and reports latency, throughput and optionally recall@k.
[[nodiscard]] Status run_search(const SearchOptions& options, std::ostream& log);

// Prints header, sections, parameters and (for HNSW) the level histogram.
[[nodiscard]] Status run_info(const std::filesystem::path& index, std::ostream& log);

// Loads with full checksum verification and checks graph invariants. Returns the first problem.
[[nodiscard]] Status run_verify(const std::filesystem::path& index, std::ostream& log);

}  // namespace vf::cli
