#ifndef ULBVC_FRAMEWORK_THREAD_POOL_H_
#define ULBVC_FRAMEWORK_THREAD_POOL_H_

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <vector>

#include "ulbvc/core/status.h"
#include "ulbvc/framework/task_executor.h"

namespace ulbvc {

class ThreadPool final : public TaskExecutor {
 public:
  explicit ThreadPool(std::size_t thread_count = 0, std::size_t max_queue_size = 0);
  ~ThreadPool() override;

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  std::future<void> SubmitTask(std::unique_ptr<Task> task) override;

  template <typename T>
  std::future<T> SubmitValue(std::function<T()> task) {
    Check(task != nullptr, "cannot submit an empty thread-pool task");
    auto promise = std::make_shared<std::promise<T>>();
    std::future<T> future = promise->get_future();
    Enqueue([task = std::move(task), promise]() mutable {
      try {
        if constexpr (std::is_void_v<T>) {
          task();
          promise->set_value();
        } else {
          promise->set_value(task());
        }
      } catch (...) {
        promise->set_exception(std::current_exception());
      }
    });
    return future;
  }

  std::size_t thread_count() const { return workers_.size(); }
  void Stop() override;

 private:
  void Enqueue(std::function<void()> task);
  void WorkerMain();

  const std::size_t max_queue_size_ = 0;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  bool stopping_ = false;
  std::queue<std::function<void()>> tasks_;
  std::vector<std::thread> workers_;
};

}  // namespace ulbvc

#endif  // ULBVC_FRAMEWORK_THREAD_POOL_H_
