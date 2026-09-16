#pragma once

// Fixed-size thread pool (docs/DESIGN.md §11.2, docs/concurrency.md).
//
// Thread safety: every member function may be called concurrently from any thread, including from
// tasks running on the pool, except that the destructor must not race with other calls.
//
// - submit()/try_submit() queue a task and return a std::future for its result; exceptions thrown
//   by the task are delivered through the future.
// - parallel_for() splits [begin, end) into chunks of about `grain` indices, runs them on the
//   workers AND the calling thread, and returns when every chunk has finished (no task outlives
//   the call). The first exception thrown by the body is rethrown in the caller after all started
//   chunks finish; chunks not yet started are skipped. Called from a pool worker (nested use) it
//   runs inline on that worker, which avoids deadlock and oversubscription.
// - shutdown() (also run by the destructor) stops accepting tasks, runs every queued task, and
//   joins the workers.

#include <cstddef>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <vectorforge/status.hpp>

namespace vf {

namespace detail {

// Move-only type-erased void() callable (std::move_only_function is C++23).
class UniqueFunction {
 public:
  UniqueFunction() noexcept = default;
  template <class F>
    requires(!std::is_same_v<std::remove_cvref_t<F>, UniqueFunction> &&
             std::is_invocable_v<std::decay_t<F>&>)
  // NOLINTNEXTLINE(google-explicit-constructor, bugprone-forwarding-reference-overload)
  UniqueFunction(F&& f) : impl_(std::make_unique<Model<std::decay_t<F>>>(std::forward<F>(f))) {}

  void operator()() { impl_->call(); }
  explicit operator bool() const noexcept { return impl_ != nullptr; }

 private:
  struct Concept {
    Concept() = default;
    Concept(const Concept&) = delete;
    Concept& operator=(const Concept&) = delete;
    Concept(Concept&&) = delete;
    Concept& operator=(Concept&&) = delete;
    virtual ~Concept() = default;
    virtual void call() = 0;
  };
  template <class F>
  struct Model final : Concept {
    explicit Model(F&& f) : fn(std::move(f)) {}
    explicit Model(const F& f) : fn(f) {}
    void call() override { std::invoke(fn); }
    F fn;
  };
  std::unique_ptr<Concept> impl_;
};

// Non-owning reference to a callable invoked as body(lo, hi).
class RangeBody {
 public:
  template <class F>
  explicit RangeBody(F& f) noexcept
      : object_(std::addressof(f)), call_([](void* object, std::size_t lo, std::size_t hi) {
          (*static_cast<F*>(object))(lo, hi);
        }) {}
  void operator()(std::size_t lo, std::size_t hi) const { call_(object_, lo, hi); }

 private:
  void* object_;
  void (*call_)(void*, std::size_t, std::size_t);
};

}  // namespace detail

class ThreadPool {
 public:
  // `threads` workers; 0 selects std::thread::hardware_concurrency() (at least 1).
  // Throws std::system_error if a thread cannot be started.
  explicit ThreadPool(std::size_t threads = 0);
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;
  ~ThreadPool();

  // Number of worker threads.
  [[nodiscard]] std::size_t size() const noexcept;

  // Queues `f`. Errors: Unavailable (after shutdown()). Throws std::bad_alloc.
  template <class F>
  [[nodiscard]] auto try_submit(F&& f)
      -> Result<std::future<std::invoke_result_t<std::decay_t<F>&>>> {
    using R = std::invoke_result_t<std::decay_t<F>&>;
    std::packaged_task<R()> task(std::forward<F>(f));
    std::future<R> future = task.get_future();
    if (!enqueue(detail::UniqueFunction(std::move(task)))) {
      return Status::unavailable("thread pool is shut down");
    }
    return future;
  }

  // As try_submit(); after shutdown() the returned future holds a std::runtime_error.
  template <class F>
  [[nodiscard]] auto submit(F&& f) -> std::future<std::invoke_result_t<std::decay_t<F>&>> {
    using R = std::invoke_result_t<std::decay_t<F>&>;
    auto result = try_submit(std::forward<F>(f));
    if (result.ok()) {
      return std::move(result).value();
    }
    std::promise<R> failed;
    failed.set_exception(std::make_exception_ptr(std::runtime_error(result.status().to_string())));
    return failed.get_future();
  }

  // Calls body(lo, hi) for disjoint ranges covering [begin, end); grain 0 is treated as 1.
  template <class Body>
  void parallel_for(std::size_t begin, std::size_t end, std::size_t grain, Body&& body) {
    run_ranges(begin, end, grain, detail::RangeBody(body));
  }

  // Stops accepting tasks, runs the queued ones and joins the workers. Idempotent.
  void shutdown() noexcept;

  // True on a thread owned by any ThreadPool.
  [[nodiscard]] static bool on_worker_thread() noexcept;

 private:
  bool enqueue(detail::UniqueFunction task);
  void run_ranges(std::size_t begin, std::size_t end, std::size_t grain, detail::RangeBody body);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vf
