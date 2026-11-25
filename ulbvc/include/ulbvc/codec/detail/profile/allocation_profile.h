#ifndef ULBVC_CODEC_DETAIL_ALLOCATION_PROFILE_H_
#define ULBVC_CODEC_DETAIL_ALLOCATION_PROFILE_H_

#include <ulbvc/codec/detail/profile/allocation_tracking.h>

#include <string>

#include "ulbvc/framework/profiler.h"

namespace ulbvc::codec {

void AddAllocationProfile(ulbvc::Profiler* profiler, const char* name,
                          const codec::AllocationSnapshot& before,
                          const codec::AllocationSnapshot& after);
void AddHotPathAllocationProfile(ulbvc::Profiler* profiler, const codec::AllocationSnapshot& before,
                                 const codec::AllocationSnapshot& after);

class ScopedAllocationSpan {
 public:
  ScopedAllocationSpan(ulbvc::Profiler* profiler, const char* name);
  ScopedAllocationSpan(const ScopedAllocationSpan&) = delete;
  ScopedAllocationSpan& operator=(const ScopedAllocationSpan&) = delete;
  ~ScopedAllocationSpan();

 private:
  ulbvc::Profiler* profiler_ = nullptr;
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

}  // namespace ulbvc::codec

#endif  // ULBVC_CODEC_DETAIL_ALLOCATION_PROFILE_H_
