#include <mlvc/codec/detail/profile/codec_profile.h>

#include <chrono>
#include <utility>

#include <mlvc/codec/detail/profile/allocation_tracking.h>
#include "mlvc/core/status.h"

namespace mlvc::codec {

AllocationSnapshot SnapshotAllocations() {
  return AllocationSnapshot{
      g_repository_allocation_count.load(std::memory_order_relaxed),
      g_repository_allocation_bytes.load(std::memory_order_relaxed),
  };
}

AllocationSnapshot SnapshotFullProcessAllocations() {
  return AllocationSnapshot{
      g_allocation_count.load(std::memory_order_relaxed),
      g_allocation_bytes.load(std::memory_order_relaxed),
  };
}

bool AllocationTrackingEnabled() { return g_track_allocations.load(std::memory_order_relaxed); }

bool RepositoryAllocationTrackingEnabled() {
  return g_track_repository_allocations.load(std::memory_order_relaxed);
}

void SetAllocationTracking(bool enabled) {
  g_track_allocations.store(enabled, std::memory_order_relaxed);
  g_track_repository_allocations.store(enabled, std::memory_order_relaxed);
}

void SetRepositoryAllocationTracking(bool enabled) {
  g_track_repository_allocations.store(enabled, std::memory_order_relaxed);
}

ScopedRepositoryAllocationTrackingPause::ScopedRepositoryAllocationTrackingPause()
    : was_repository_tracking_(RepositoryAllocationTrackingEnabled()) {
  SetRepositoryAllocationTracking(false);
}

ScopedRepositoryAllocationTrackingPause::~ScopedRepositoryAllocationTrackingPause() {
  SetRepositoryAllocationTracking(was_repository_tracking_);
}

ScopedProfileTimer::ScopedProfileTimer(mlvc::Profiler* profiler, const char* name)
    : profiler_(profiler), name_(name), begin_(std::chrono::steady_clock::now()) {}

ScopedProfileTimer::~ScopedProfileTimer() {
  if (profiler_ == nullptr) {
    return;
  }
  const auto end = std::chrono::steady_clock::now();
  AddProfileEvent(profiler_, name_, profiler_->StartMs(begin_), profiler_->DurationMs(begin_, end));
}

void AddProfileEvent(mlvc::Profiler* profiler, const char* name, double start_ms,
                     double duration_ms) {
  if (profiler == nullptr) {
    return;
  }
  ScopedRepositoryAllocationTrackingPause allocation_pause;
  profiler->AddEvent(name, start_ms, duration_ms);
}

void AddProfileEventWithArgs(mlvc::Profiler* profiler, std::string name, std::string category,
                             std::string thread, double start_ms, double duration_ms,
                             std::vector<mlvc::ProfileArgument> args) {
  if (profiler == nullptr) {
    return;
  }
  ScopedRepositoryAllocationTrackingPause allocation_pause;
  profiler->AddEventWithArgs(std::move(name), std::move(category), std::move(thread), start_ms,
                             duration_ms, std::move(args));
}

void AddProfileDuration(mlvc::Profiler* profiler, const char* name,
                        std::chrono::steady_clock::time_point begin,
                        std::chrono::steady_clock::time_point end) {
  if (profiler == nullptr) {
    return;
  }
  ScopedRepositoryAllocationTrackingPause allocation_pause;
  profiler->AddDurationNow(name, begin, end);
}

}  // namespace mlvc::codec
