#pragma once

// `vectorforge` CLI commands, callable in-process (the executable only parses arguments).

#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <memory>
#include <optional>
#include <string>

#include <vectorforge/config.hpp>
#include <vectorforge/status.hpp>
#include <vectorforge/thread_pool.hpp>
#include <vectorforge/types.hpp>

#include "util/synthetic.hpp"

#if defined(VF_HAVE_SERVER)
#include "server/server.hpp"
#endif

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
  std::uint32_t threads = 0;  // search threads including the caller; 0 = all hardware threads
};

// Exact k-NN ground truth with a Flat collection (queries in parallel).
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
  std::uint32_t threads = 0;  // worker threads including the caller; 0 = all hardware threads
};

// Builds a collection from a dataset file and saves it.
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

#if defined(VF_HAVE_SERVER)
struct ServeOptions {
  std::filesystem::path data_dir;
  server::ServerConfig server;
  bool use_mmap = true;
  bool snapshot_on_exit = false;
  bool install_signal_handlers = true;
};

// Serves `data_dir` until SIGINT/SIGTERM or request_serve_stop(); then drains, stops and
// optionally snapshots every collection.
[[nodiscard]] Status run_serve(const ServeOptions& options, std::ostream& log);
void request_serve_stop() noexcept;
#endif

// Pool giving `threads` threads together with the calling thread (0 = hardware concurrency);
// nullptr when one thread is requested.
[[nodiscard]] std::unique_ptr<ThreadPool> make_pool(std::uint32_t threads);

}  // namespace vf::cli
