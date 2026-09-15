#ifndef MLVC_FRAMEWORK_PROFILER_H_
#define MLVC_FRAMEWORK_PROFILER_H_

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace mlvc {

enum class ProfileArgType {
  kString,
  kNumber,
  kBool,
};

struct ProfileArgument {
  std::string key;
  std::string value;
  ProfileArgType type = ProfileArgType::kString;
};

struct ProfileEvent {
  std::string name;
  std::string category;
  std::string thread;
  double start_ms = 0.0;
  double duration_ms = 0.0;
  bool counter = false;
  std::vector<ProfileArgument> args;
};

class Profiler {
 public:
  Profiler();

  void Reset();
  void ReserveEvents(std::size_t capacity);
  void AddEvent(const std::string& name, double start_ms, double duration_ms);
  void AddEvent(const char* name, double start_ms, double duration_ms);
  void AddEventWithArgs(std::string name, std::string category, std::string thread, double start_ms,
                        double duration_ms, std::vector<ProfileArgument> args);
  void AddCounter(std::string name, std::string category, std::string thread, double start_ms,
                  std::vector<ProfileArgument> args);
  void AddDurationNow(const std::string& name, std::chrono::steady_clock::time_point begin,
                      std::chrono::steady_clock::time_point end);
  void AddDurationNow(const char* name, std::chrono::steady_clock::time_point begin,
                      std::chrono::steady_clock::time_point end);
  double StartMs(std::chrono::steady_clock::time_point time) const;
  double DurationMs(std::chrono::steady_clock::time_point begin,
                    std::chrono::steady_clock::time_point end) const;
  const std::vector<ProfileEvent>& events() const { return events_; }
  std::map<std::string, double> TotalsByName() const;
  void WriteChromeTrace(const std::filesystem::path& path) const;

  static ProfileArgument Arg(std::string key, std::string value);
  static ProfileArgument Arg(std::string key, const char* value);
  static ProfileArgument Arg(std::string key, uint64_t value);
  static ProfileArgument Arg(std::string key, int64_t value);
  static ProfileArgument Arg(std::string key, int value);
  static ProfileArgument Arg(std::string key, double value);
  static ProfileArgument BoolArg(std::string key, bool value);

 private:
  static std::string InferCategory(const std::string& name);
  static std::string InferThread(const std::string& name);

  std::chrono::steady_clock::time_point origin_;
  std::vector<ProfileEvent> events_;
  mutable std::mutex mutex_;
};

class ScopedCpuTimer {
 public:
  ScopedCpuTimer(Profiler* profiler, std::string name);
  ~ScopedCpuTimer();

 private:
  Profiler* profiler_ = nullptr;
  std::string name_;
  std::chrono::steady_clock::time_point begin_;
};

}  // namespace mlvc

#endif  // MLVC_FRAMEWORK_PROFILER_H_
