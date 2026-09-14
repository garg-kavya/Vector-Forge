// Runs LLVMFuzzerTestOneInput over files and directories given on the command line (corpus
// replay without libFuzzer), plus mutated variants of each file: truncations and bit flips.
// Exit code 0 means every input was processed without crashing.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size);

namespace {

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(std::filesystem::file_size(path)));
  std::ifstream in(path, std::ios::binary);
  in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  return bytes;
}

std::size_t replay(const std::vector<std::uint8_t>& input) {
  std::size_t runs = 0;
  LLVMFuzzerTestOneInput(input.data(), input.size());
  ++runs;
  // Deterministic cheap mutations: prefixes and single-bit flips at a stride.
  const std::size_t stride = input.size() / 97 + 1;
  std::vector<std::uint8_t> mutated = input;
  for (std::size_t i = 0; i < input.size(); i += stride) {
    LLVMFuzzerTestOneInput(input.data(), i);
    mutated[i] ^= static_cast<std::uint8_t>(1U << (i % 8));
    LLVMFuzzerTestOneInput(mutated.data(), mutated.size());
    mutated[i] = input[i];
    runs += 2;
  }
  return runs;
}

}  // namespace

int main(int argc, char** argv) {
  std::size_t files = 0;
  std::size_t runs = 0;
  for (int a = 1; a < argc; ++a) {
    const std::filesystem::path arg(argv[a]);
    std::vector<std::filesystem::path> paths;
    if (std::filesystem::is_directory(arg)) {
      for (const auto& entry : std::filesystem::directory_iterator(arg)) {
        if (entry.is_regular_file()) {
          paths.push_back(entry.path());
        }
      }
    } else {
      paths.push_back(arg);
    }
    for (const auto& path : paths) {
      runs += replay(read_file(path));
      ++files;
    }
  }
  std::cout << "replayed " << files << " files, " << runs << " inputs\n";
  return files > 0 ? 0 : 1;
}
