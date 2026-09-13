#include <cstdint>
#include <fstream>
#include <ostream>
#include <span>
#include <string>
#include <vector>

#include "commands.hpp"
#include "util/dataset_io.hpp"
#include "util/synthetic.hpp"
#include "util/timer.hpp"

namespace vf::cli {

Status run_gen_data(const GenDataOptions& options, std::ostream& log) {
  if (options.rows == 0) {
    return Status::invalid_argument("--n must be at least 1");
  }
  if (options.out.empty()) {
    return Status::invalid_argument("--out is required");
  }
  Result<detail::SyntheticGenerator> generator = detail::SyntheticGenerator::create(options.spec);
  if (!generator.ok()) {
    return generator.status();
  }
  detail::SyntheticGenerator& gen = generator.value();
  const std::uint32_t dim = options.spec.dim;
  std::vector<float> row(dim);
  const detail::Stopwatch timer;

  if (options.format == DatasetFormat::Npy) {
    Result<detail::NpyWriter> writer =
        detail::NpyWriter::create(options.out, detail::NpyDtype::Float32, options.rows, dim);
    if (!writer.ok()) {
      return writer.status();
    }
    for (std::uint64_t r = 0; r < options.rows; ++r) {
      gen.next_row(row);
      VF_RETURN_IF_ERROR(writer.value().write(std::span<const float>(row)));
    }
    VF_RETURN_IF_ERROR(writer.value().finish());
  } else {
    std::ofstream out(options.out, std::ios::binary | std::ios::trunc);
    if (!out) {
      return Status::io_error("cannot create " + options.out.string());
    }
    const auto dim_i32 = static_cast<std::int32_t>(dim);
    for (std::uint64_t r = 0; r < options.rows; ++r) {
      gen.next_row(row);
      out.write(reinterpret_cast<const char*>(&dim_i32), sizeof(dim_i32));
      out.write(reinterpret_cast<const char*>(row.data()),
                static_cast<std::streamsize>(row.size() * sizeof(float)));
    }
    out.flush();
    if (!out) {
      return Status::io_error("failed writing " + options.out.string());
    }
  }

  log << "wrote " << options.rows << " x " << dim << " float32 vectors to " << options.out.string()
      << " in " << timer.elapsed_seconds() << " s\n";
  return {};
}

}  // namespace vf::cli
