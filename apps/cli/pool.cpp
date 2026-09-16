#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>

#include <vectorforge/thread_pool.hpp>

#include "commands.hpp"

namespace vf::cli {

std::unique_ptr<ThreadPool> make_pool(std::uint32_t threads) {
  const std::size_t total =
      threads == 0 ? std::max<std::size_t>(1, std::thread::hardware_concurrency()) : threads;
  if (total <= 1) {
    return nullptr;
  }
  return std::make_unique<ThreadPool>(total - 1);  // the calling thread takes part
}

}  // namespace vf::cli
