#ifndef MLVC_APPLICATION_STREAM_ENCODE_ENCODE_SCHEDULE_H_
#define MLVC_APPLICATION_STREAM_ENCODE_ENCODE_SCHEDULE_H_

#include <mlvc/core/status.h>

#include <cstddef>
#include <future>

namespace mlvc::codec {

// Early writes also advance rate-control feedback. Restrict opportunistic
// retirement to fixed-QP prefetch so adaptive-QP decisions retain their order.
inline bool ShouldRetireReadyEntropy(int prefetch_frames, bool rate_control_enabled,
                                    std::future_status front_status) {
  return prefetch_frames > 0 && !rate_control_enabled &&
         front_status == std::future_status::ready;
}

inline bool ShouldRetirePendingEntropy(std::size_t pending_size, std::size_t capacity) {
  mlvc::Check(capacity > 0, "encode pending entropy capacity must be positive");
  return pending_size > capacity;
}

}  // namespace mlvc::codec

#endif  // MLVC_APPLICATION_STREAM_ENCODE_ENCODE_SCHEDULE_H_
