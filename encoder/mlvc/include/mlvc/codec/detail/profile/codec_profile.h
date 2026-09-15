#ifndef MLVC_CODEC_DETAIL_CODEC_PROFILE_H_
#define MLVC_CODEC_DETAIL_CODEC_PROFILE_H_

#include <chrono>
#include <string>
#include <vector>

#include "mlvc/framework/profiler.h"

namespace mlvc::codec {

void AddProfileEvent(mlvc::Profiler* profiler, const char* name, double start_ms,
                     double duration_ms);
void AddProfileEventWithArgs(mlvc::Profiler* profiler, std::string name, std::string category,
                             std::string thread, double start_ms, double duration_ms,
                             std::vector<mlvc::ProfileArgument> args);
void AddProfileDuration(mlvc::Profiler* profiler, const char* name,
                        std::chrono::steady_clock::time_point begin,
                        std::chrono::steady_clock::time_point end);

class ScopedProfileTimer {
 public:
  ScopedProfileTimer(mlvc::Profiler* profiler, const char* name);

  ScopedProfileTimer(const ScopedProfileTimer&) = delete;
  ScopedProfileTimer& operator=(const ScopedProfileTimer&) = delete;

  ~ScopedProfileTimer();

 private:
  mlvc::Profiler* profiler_ = nullptr;
  const char* name_ = "";
  std::chrono::steady_clock::time_point begin_;
};

}  // namespace mlvc::codec

#endif  // MLVC_CODEC_DETAIL_CODEC_PROFILE_H_
