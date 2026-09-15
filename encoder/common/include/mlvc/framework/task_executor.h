#ifndef MLVC_FRAMEWORK_TASK_EXECUTOR_H_
#define MLVC_FRAMEWORK_TASK_EXECUTOR_H_

#include <functional>
#include <future>
#include <memory>
#include <string>
#include <utility>

namespace mlvc {

class Task {
 public:
  virtual ~Task() = default;
  virtual void Run() = 0;
  virtual const std::string& name() const = 0;
  virtual int frame_index() const = 0;
};

class FunctionTask final : public Task {
 public:
  FunctionTask(std::string name, int frame_index, std::function<void()> function)
      : name_(std::move(name)), frame_index_(frame_index), function_(std::move(function)) {}

  void Run() override { function_(); }
  const std::string& name() const override { return name_; }
  int frame_index() const override { return frame_index_; }

 private:
  std::string name_;
  int frame_index_ = -1;
  std::function<void()> function_;
};

class TaskExecutor {
 public:
  virtual ~TaskExecutor() = default;
  virtual std::future<void> SubmitTask(std::unique_ptr<Task> task) = 0;
  virtual void Stop() = 0;
};

}  // namespace mlvc

#endif  // MLVC_FRAMEWORK_TASK_EXECUTOR_H_
