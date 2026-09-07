#ifndef MLVC_CODEC_DETAIL_ALLOCATION_PROFILE_H_
#define MLVC_CODEC_DETAIL_ALLOCATION_PROFILE_H_

#include <string>

#include <mlvc/codec/detail/profile/allocation_tracking.h>
#include "mlvc/framework/profiler.h"

namespace mlvc::codec {

void AddAllocationProfile(mlvc::Profiler* profiler, const char* name,
                          const codec::AllocationSnapshot& before,
                          const codec::AllocationSnapshot& after);
void AddHotPathAllocationProfile(mlvc::Profiler* profiler, const codec::AllocationSnapshot& before,
                                 const codec::AllocationSnapshot& after);

class ScopedAllocationSpan {
 public:
  ScopedAllocationSpan(mlvc::Profiler* profiler, const char* name);
  ScopedAllocationSpan(const ScopedAllocationSpan&) = delete;
  ScopedAllocationSpan& operator=(const ScopedAllocationSpan&) = delete;
  ~ScopedAllocationSpan();

 private:
  mlvc::Profiler* profiler_ = nullptr;
  const char* name_ = "";
  codec::AllocationSnapshot before_;
};

class ScopedAllocationTrackingPause {
 public:
  ScopedAllocationTrackingPause();
  ScopedAllocationTrackingPause(const ScopedAllocationTrackingPause&) = delete;
  ScopedAllocationTrackingPause& operator=(const ScopedAllocationTrackingPause&) = delete;
  ~ScopedAllocationTrackingPause();

 private:
  bool was_tracking_ = false;
  bool was_repository_tracking_ = false;
};

}  // namespace mlvc::codec

#endif  // MLVC_CODEC_DETAIL_ALLOCATION_PROFILE_H_
