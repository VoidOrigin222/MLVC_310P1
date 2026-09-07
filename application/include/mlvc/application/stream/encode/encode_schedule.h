#ifndef MLVC_APPLICATION_STREAM_ENCODE_ENCODE_SCHEDULE_H_
#define MLVC_APPLICATION_STREAM_ENCODE_ENCODE_SCHEDULE_H_

#include <mlvc/core/status.h>

#include <cstddef>

namespace mlvc::codec {

inline bool ShouldRetirePendingEntropy(std::size_t pending_size, std::size_t capacity) {
  mlvc::Check(capacity > 0, "encode pending entropy capacity must be positive");
  return pending_size > capacity;
}

}  // namespace mlvc::codec

#endif  // MLVC_APPLICATION_STREAM_ENCODE_ENCODE_SCHEDULE_H_
