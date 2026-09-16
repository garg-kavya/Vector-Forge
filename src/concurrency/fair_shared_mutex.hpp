#pragma once

// Reader/writer lock with bounded waiting on both sides (docs/concurrency.md "Fairness").
//
// std::shared_mutex makes no fairness promise, and on Windows (SRWLOCK) it was measured to starve
// searches behind a writer that re-locks between add_batch slices, and to starve that writer behind
// a stream of searches. This lock alternates:
//   - while a writer waits, new readers wait (writer preference), so a writer waits at most for the
//     readers that already hold the lock;
//   - when a writer unlocks, every reader waiting at that moment is admitted before the next
//     writer, so a reader waits at most for one writer section.
// Meets the SharedMutex requirements used by std::shared_lock / std::unique_lock.

#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace vf::detail {

class FairSharedMutex {
 public:
  FairSharedMutex() = default;
  FairSharedMutex(const FairSharedMutex&) = delete;
  FairSharedMutex& operator=(const FairSharedMutex&) = delete;
  FairSharedMutex(FairSharedMutex&&) = delete;
  FairSharedMutex& operator=(FairSharedMutex&&) = delete;
  ~FairSharedMutex() = default;

  void lock_shared() {
    std::unique_lock<std::mutex> lock(mutex_);
    ++waiting_readers_;
    readers_cv_.wait(
        lock, [this] { return !writer_active_ && (waiting_writers_ == 0 || reader_quota_ > 0); });
    --waiting_readers_;
    if (reader_quota_ > 0) {
      --reader_quota_;
    }
    ++active_readers_;
  }

  void unlock_shared() {
    const std::lock_guard<std::mutex> lock(mutex_);
    --active_readers_;
    if (active_readers_ == 0 && waiting_writers_ > 0) {
      writers_cv_.notify_one();
    }
  }

  void lock() {
    std::unique_lock<std::mutex> lock(mutex_);
    ++waiting_writers_;
    writers_cv_.wait(
        lock, [this] { return !writer_active_ && active_readers_ == 0 && reader_quota_ == 0; });
    --waiting_writers_;
    writer_active_ = true;
  }

  void unlock() {
    const std::lock_guard<std::mutex> lock(mutex_);
    writer_active_ = false;
    // Admit the readers that queued behind this writer before any further writer.
    reader_quota_ = waiting_readers_;
    if (waiting_readers_ > 0) {
      readers_cv_.notify_all();
    } else if (waiting_writers_ > 0) {
      writers_cv_.notify_one();
    }
  }

 private:
  std::mutex mutex_;
  std::condition_variable readers_cv_;
  std::condition_variable writers_cv_;
  std::uint64_t active_readers_ = 0;
  std::uint64_t waiting_readers_ = 0;
  std::uint64_t waiting_writers_ = 0;
  std::uint64_t reader_quota_ = 0;  // readers still admitted ahead of waiting writers
  bool writer_active_ = false;
};

}  // namespace vf::detail
