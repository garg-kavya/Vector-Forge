#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <vectorforge/thread_pool.hpp>

#include "core/assert.hpp"

namespace vf {

namespace {

// Pool whose worker is running on this thread (nullptr elsewhere).
thread_local const void* tls_pool = nullptr;

}  // namespace

struct ThreadPool::Impl {
  std::size_t thread_count = 0;
  std::mutex join_mutex;  // serialises shutdown()
  std::mutex mutex;
  std::condition_variable wake;
  std::deque<detail::UniqueFunction> queue;
  bool stopping = false;
  std::vector<std::jthread> workers;

  void work() {
    tls_pool = this;
    for (;;) {
      detail::UniqueFunction task;
      {
        std::unique_lock<std::mutex> lock(mutex);
        wake.wait(lock, [this] { return stopping || !queue.empty(); });
        if (queue.empty()) {
          return;  // stopping and drained
        }
        task = std::move(queue.front());
        queue.pop_front();
      }
      // Tasks are packaged_tasks or parallel_for helpers; both capture their own exceptions.
      task();
    }
  }
};

ThreadPool::ThreadPool(std::size_t threads) : impl_(std::make_unique<Impl>()) {
  if (threads == 0) {
    threads = std::max<std::size_t>(1, std::thread::hardware_concurrency());
  }
  impl_->thread_count = threads;
  impl_->workers.reserve(threads);
  try {
    for (std::size_t i = 0; i < threads; ++i) {
      impl_->workers.emplace_back([impl = impl_.get()] { impl->work(); });
    }
  } catch (...) {
    shutdown();
    throw;
  }
}

ThreadPool::~ThreadPool() {
  shutdown();
}

std::size_t ThreadPool::size() const noexcept {
  return impl_->thread_count;
}

bool ThreadPool::on_worker_thread() noexcept {
  return tls_pool != nullptr;
}

void ThreadPool::shutdown() noexcept {
  // A worker cannot join itself; calling shutdown() (or destroying the pool) from one of its own
  // tasks is a precondition violation.
  VF_CHECK(tls_pool != impl_.get(), "ThreadPool::shutdown called from one of its own workers");
  const std::lock_guard<std::mutex> join_lock(impl_->join_mutex);
  std::vector<std::jthread> workers;
  {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->stopping = true;
    workers.swap(impl_->workers);
  }
  impl_->wake.notify_all();
  for (std::jthread& worker : workers) {
    worker.join();  // workers drain the queue before exiting
  }
}

bool ThreadPool::enqueue(detail::UniqueFunction task) {
  {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->stopping) {
      return false;
    }
    impl_->queue.push_back(std::move(task));
  }
  impl_->wake.notify_one();
  return true;
}

void ThreadPool::run_ranges(std::size_t begin, std::size_t end, std::size_t grain,
                            detail::RangeBody body) {
  if (begin >= end) {
    return;
  }
  const std::size_t count = end - begin;
  grain = std::max<std::size_t>(grain, 1);
  const std::size_t chunks = (count / grain) + (count % grain != 0 ? 1 : 0);
  if (chunks == 1 || on_worker_thread()) {
    body(begin, end);
    return;
  }

  struct Shared {
    std::atomic<std::size_t> next{0};
    std::atomic<bool> failed{false};
    std::mutex mutex;
    std::condition_variable done;
    std::size_t running_helpers = 0;
    std::exception_ptr error;
  };
  // Helpers keep the shared state alive until they are completely done, even after the caller has
  // observed running_helpers == 0 and returned.
  const auto state = std::make_shared<Shared>();
  Shared& shared = *state;

  auto run_chunks = [&shared, &body, begin, end, grain, chunks] {
    for (;;) {
      if (shared.failed.load(std::memory_order_relaxed)) {
        return;
      }
      const std::size_t chunk = shared.next.fetch_add(1, std::memory_order_relaxed);
      if (chunk >= chunks) {
        return;
      }
      const std::size_t lo = begin + (chunk * grain);
      const std::size_t hi = std::min(end, lo + grain);
      try {
        body(lo, hi);
      } catch (...) {
        const std::lock_guard<std::mutex> lock(shared.mutex);
        if (!shared.error) {
          shared.error = std::current_exception();
        }
        shared.failed.store(true, std::memory_order_relaxed);
        return;
      }
    }
  };

  // One helper per worker at most; the caller takes part too.
  const std::size_t helpers = std::min(chunks - 1, size());
  for (std::size_t h = 0; h < helpers; ++h) {
    {
      const std::lock_guard<std::mutex> lock(shared.mutex);
      ++shared.running_helpers;
    }
    bool queued = false;
    try {
      queued = enqueue([state, &run_chunks] {
        run_chunks();  // the caller waits for this helper, so run_chunks and body are alive
        const std::lock_guard<std::mutex> lock(state->mutex);
        --state->running_helpers;
        state->done.notify_all();
      });
    } catch (...) {
      queued = false;  // allocation failure: the caller runs the remaining chunks
    }
    if (!queued) {
      const std::lock_guard<std::mutex> lock(shared.mutex);
      --shared.running_helpers;
      break;
    }
  }

  run_chunks();
  std::unique_lock<std::mutex> lock(shared.mutex);
  shared.done.wait(lock, [&shared] { return shared.running_helpers == 0; });
  if (shared.error) {
    std::rethrow_exception(shared.error);
  }
}

}  // namespace vf
