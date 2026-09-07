#include "mlvc/framework/thread_pool.h"

#include <algorithm>
#include <utility>

namespace mlvc {

ThreadPool::ThreadPool(std::size_t thread_count, std::size_t max_queue_size)
    : max_queue_size_(max_queue_size) {
  if (thread_count == 0) {
    thread_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
  }
  workers_.reserve(thread_count);
  for (std::size_t index = 0; index < thread_count; ++index) {
    workers_.emplace_back(&ThreadPool::WorkerMain, this);
  }
}

ThreadPool::~ThreadPool() { Stop(); }

void ThreadPool::Enqueue(std::function<void()> task) {
  Check(task != nullptr, "cannot enqueue an empty thread-pool task");
  {
    std::lock_guard<std::mutex> lock(mutex_);
    Check(!stopping_, "cannot submit task after thread pool stop");
    Check(max_queue_size_ == 0 || tasks_.size() < max_queue_size_,
          "thread-pool queue is full");
    tasks_.push(std::move(task));
  }
  condition_.notify_one();
}

std::future<void> ThreadPool::SubmitTask(std::unique_ptr<Task> task) {
  Check(task != nullptr, "cannot submit a null task");
  std::shared_ptr<Task> shared_task = std::move(task);
  auto promise = std::make_shared<std::promise<void>>();
  std::future<void> future = promise->get_future();
  Enqueue([task = std::move(shared_task), promise]() mutable {
    try {
      task->Run();
      promise->set_value();
    } catch (...) {
      promise->set_exception(std::current_exception());
    }
  });
  return future;
}

void ThreadPool::Stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_) {
      return;
    }
    stopping_ = true;
  }
  condition_.notify_all();
  for (std::thread& worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

void ThreadPool::WorkerMain() {
  for (;;) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      condition_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
      if (stopping_ && tasks_.empty()) {
        return;
      }
      task = std::move(tasks_.front());
      tasks_.pop();
    }
    task();
  }
}

}  // namespace mlvc
