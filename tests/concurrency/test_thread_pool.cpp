// ThreadPool (docs/DESIGN.md §11.2, §11.7): results, exception propagation, range coverage,
// nested parallel_for, shutdown with queued tasks, many tiny tasks. Meaningful under TSan.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <future>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <vectorforge/thread_pool.hpp>

namespace {

using vf::ThreadPool;

TEST(ThreadPool, SizeDefaultsToHardwareConcurrency) {
  const ThreadPool pool;
  EXPECT_GE(pool.size(), 1U);
  EXPECT_EQ(pool.size(), std::max<std::size_t>(1, std::thread::hardware_concurrency()));
  const ThreadPool three(3);
  EXPECT_EQ(three.size(), 3U);
}

TEST(ThreadPool, SubmitReturnsValuesAndExceptions) {
  ThreadPool pool(4);
  std::future<int> value = pool.submit([] { return 42; });
  std::future<void> failing = pool.submit([] { throw std::runtime_error("boom"); });
  // Move-only callables are accepted.
  auto owned = std::make_unique<int>(7);
  std::future<int> moved = pool.submit([p = std::move(owned)] { return *p; });
  EXPECT_EQ(value.get(), 42);
  EXPECT_EQ(moved.get(), 7);
  EXPECT_THROW(failing.get(), std::runtime_error);
  EXPECT_FALSE(ThreadPool::on_worker_thread());
  EXPECT_TRUE(pool.submit([] { return ThreadPool::on_worker_thread(); }).get());
}

TEST(ThreadPool, ParallelForCoversEveryIndexExactlyOnce) {
  ThreadPool pool(4);
  for (const std::size_t n : {0U, 1U, 2U, 7U, 64U, 1000U, 4097U}) {
    for (const std::size_t grain : {0U, 1U, 3U, 16U, 5000U}) {
      std::vector<std::atomic<int>> hits(n + 5);
      pool.parallel_for(5, 5 + n, grain, [&](std::size_t lo, std::size_t hi) {
        ASSERT_LE(lo, hi);
        for (std::size_t i = lo; i < hi; ++i) {
          hits[i].fetch_add(1);
        }
      });
      for (std::size_t i = 0; i < hits.size(); ++i) {
        EXPECT_EQ(hits[i].load(), i >= 5 ? 1 : 0) << "n=" << n << " grain=" << grain << " i=" << i;
      }
    }
  }
}

TEST(ThreadPool, ParallelForRethrowsAfterEveryStartedChunkFinished) {
  ThreadPool pool(4);
  std::atomic<int> running{0};
  std::atomic<int> started{0};
  try {
    pool.parallel_for(0, 1000, 1, [&](std::size_t lo, std::size_t /*hi*/) {
      ++running;
      ++started;
      std::this_thread::sleep_for(std::chrono::microseconds(50));
      --running;
      if (lo == 3) {
        throw std::invalid_argument("chunk 3");
      }
    });
    FAIL() << "expected an exception";
  } catch (const std::invalid_argument& e) {
    EXPECT_EQ(std::string(e.what()), "chunk 3");
  }
  EXPECT_EQ(running.load(), 0) << "a chunk was still running after parallel_for returned";
  EXPECT_LT(started.load(), 1000) << "chunks after the failure should be skipped";
  // The pool stays usable.
  EXPECT_EQ(pool.submit([] { return 1; }).get(), 1);
}

TEST(ThreadPool, CallerParticipates) {
  ThreadPool pool(1);
  const auto caller = std::this_thread::get_id();
  std::atomic<bool> caller_ran{false};
  // Occupy the only worker so the caller has to start chunks itself; the caller then releases the
  // worker so the queued helper can finish.
  std::promise<void> release;
  std::future<void> blocker = pool.submit([f = release.get_future().share()] { f.wait(); });
  std::once_flag released;
  pool.parallel_for(0, 8, 1, [&](std::size_t, std::size_t) {
    if (std::this_thread::get_id() == caller) {
      caller_ran = true;
      std::call_once(released, [&] { release.set_value(); });
    }
  });
  blocker.get();
  EXPECT_TRUE(caller_ran.load());
}

TEST(ThreadPool, NestedParallelForRunsInline) {
  ThreadPool pool(2);
  std::atomic<std::size_t> total{0};
  pool.parallel_for(0, 8, 1, [&](std::size_t, std::size_t) {
    const auto outer = std::this_thread::get_id();
    const bool on_worker = ThreadPool::on_worker_thread();
    pool.parallel_for(0, 100, 1, [&](std::size_t lo, std::size_t hi) {
      if (on_worker) {
        EXPECT_EQ(std::this_thread::get_id(), outer) << "nested use on a worker must run inline";
      }
      total += hi - lo;
    });
  });
  EXPECT_EQ(total.load(), 800U);
}

TEST(ThreadPool, ManyTinyTasks) {
  ThreadPool pool(8);
  constexpr std::size_t kTasks = 100000;
  std::atomic<std::size_t> sum{0};
  std::vector<std::future<void>> futures;
  futures.reserve(kTasks);
  for (std::size_t i = 0; i < kTasks; ++i) {
    futures.push_back(pool.submit([&sum, i] { sum += i; }));
  }
  for (auto& f : futures) {
    f.get();
  }
  EXPECT_EQ(sum.load(), kTasks * (kTasks - 1) / 2);
}

TEST(ThreadPool, ShutdownRunsQueuedTasksThenRejects) {
  ThreadPool pool(2);
  std::atomic<int> ran{0};
  std::vector<std::future<void>> futures;
  for (int i = 0; i < 500; ++i) {
    futures.push_back(pool.submit([&ran] {
      std::this_thread::sleep_for(std::chrono::microseconds(10));
      ++ran;
    }));
  }
  pool.shutdown();
  EXPECT_EQ(ran.load(), 500);
  for (auto& f : futures) {
    EXPECT_NO_THROW(f.get());
  }
  pool.shutdown();  // idempotent

  auto rejected = pool.try_submit([] { return 1; });
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.status().code(), vf::ErrorCode::Unavailable);
  EXPECT_THROW(pool.submit([] { return 1; }).get(), std::runtime_error);

  // parallel_for still works on a shut-down pool: the caller runs every chunk.
  std::size_t covered = 0;
  pool.parallel_for(0, 50, 5, [&](std::size_t lo, std::size_t hi) { covered += hi - lo; });
  EXPECT_EQ(covered, 50U);
}

TEST(ThreadPool, ConcurrentSubmittersAndParallelFors) {
  ThreadPool pool(4);
  std::vector<std::thread> clients;
  std::atomic<std::size_t> total{0};
  for (int t = 0; t < 6; ++t) {
    clients.emplace_back([&] {
      for (int round = 0; round < 50; ++round) {
        std::vector<std::size_t> values(200);
        pool.parallel_for(0, values.size(), 7, [&](std::size_t lo, std::size_t hi) {
          for (std::size_t i = lo; i < hi; ++i) {
            values[i] = i;
          }
        });
        total += std::accumulate(values.begin(), values.end(), std::size_t{0});
        total += pool.submit([] { return std::size_t{1}; }).get();
      }
    });
  }
  for (auto& c : clients) {
    c.join();
  }
  EXPECT_EQ(total.load(), 6U * 50U * (19900U + 1U));
}

}  // namespace
