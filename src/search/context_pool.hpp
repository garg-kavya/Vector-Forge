#pragma once

// Pool of reusable SearchContexts (docs/DESIGN.md §9.5, §11.5).
//
// acquire() hands out a context exclusively through an RAII Lease and returns it on destruction.
// The pool is protected by a mutex so that const searches may run concurrently, as the
// thread-compatible Collection contract allows. A new context is created only when every pooled
// context is leased; releasing never allocates because capacity for every created context is
// reserved when it is created.

#include <cstddef>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "search/search_context.hpp"

namespace vf::detail {

class ContextPool {
 public:
  class Lease {
   public:
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    Lease(Lease&& other) noexcept
        : pool_(std::exchange(other.pool_, nullptr)), context_(std::move(other.context_)) {}
    Lease& operator=(Lease&&) = delete;
    ~Lease() {
      if (pool_ != nullptr && context_) {
        pool_->release(std::move(context_));
      }
    }

    [[nodiscard]] SearchContext& operator*() const noexcept { return *context_; }
    [[nodiscard]] SearchContext* operator->() const noexcept { return context_.get(); }

   private:
    friend class ContextPool;
    Lease(ContextPool* pool, std::unique_ptr<SearchContext> context) noexcept
        : pool_(pool), context_(std::move(context)) {}

    ContextPool* pool_;
    std::unique_ptr<SearchContext> context_;
  };

  ContextPool() = default;
  ContextPool(const ContextPool&) = delete;
  ContextPool& operator=(const ContextPool&) = delete;
  ContextPool(ContextPool&&) = delete;
  ContextPool& operator=(ContextPool&&) = delete;
  ~ContextPool() = default;

  // Throws std::bad_alloc only when a new context has to be created.
  [[nodiscard]] Lease acquire() {
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      if (!free_.empty()) {
        std::unique_ptr<SearchContext> context = std::move(free_.back());
        free_.pop_back();
        return {this, std::move(context)};
      }
    }
    auto context = std::make_unique<SearchContext>();
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      free_.reserve(created_ + 1);
      ++created_;
    }
    return {this, std::move(context)};
  }

  // Number of contexts ever created (leased or idle).
  [[nodiscard]] std::size_t created() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return created_;
  }

 private:
  void release(std::unique_ptr<SearchContext> context) noexcept {
    const std::lock_guard<std::mutex> lock(mutex_);
    free_.push_back(std::move(context));  // capacity reserved in acquire(): cannot allocate
  }

  mutable std::mutex mutex_;
  std::vector<std::unique_ptr<SearchContext>> free_;
  std::size_t created_ = 0;
};

}  // namespace vf::detail
