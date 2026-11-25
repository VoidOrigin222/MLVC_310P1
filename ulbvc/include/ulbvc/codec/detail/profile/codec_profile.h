#ifndef ULBVC_CODEC_DETAIL_CODEC_PROFILE_H_
#define ULBVC_CODEC_DETAIL_CODEC_PROFILE_H_

#include <chrono>
#include <string>
#include <vector>

#include "ulbvc/framework/profiler.h"

namespace ulbvc::codec {

void AddProfileEvent(ulbvc::Profiler* profiler, const char* name, double start_ms,
                     double duration_ms);
void AddProfileEventWithArgs(ulbvc::Profiler* profiler, std::string name, std::string category,
                             std::string thread, double start_ms, double duration_ms,
                             std::vector<ulbvc::ProfileArgument> args);
void AddProfileDuration(ulbvc::Profiler* profiler, const char* name,
                        std::chrono::steady_clock::time_point begin,
                        std::chrono::steady_clock::time_point end);

class ScopedProfileTimer {
 public:
  ScopedProfileTimer(ulbvc::Profiler* profiler, const char* name);

  ScopedProfileTimer(const ScopedProfileTimer&) = delete;
  ScopedProfileTimer& operator=(const ScopedProfileTimer&) = delete;

  ~ScopedProfileTimer();

 private:
  ulbvc::Profiler* profiler_ = nullptr;
  const char* name_ = "";
  std::chrono::steady_clock::time_point begin_;
};

}  // namespace ulbvc::codec

#endif  // ULBVC_CODEC_DETAIL_CODEC_PROFILE_H_
