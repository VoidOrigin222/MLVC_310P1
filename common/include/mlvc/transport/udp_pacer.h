#ifndef MLVC_TRANSPORT_UDP_PACER_H_
#define MLVC_TRANSPORT_UDP_PACER_H_

#include <chrono>
#include <cstddef>
#include <cstdint>

namespace mlvc::transport {

// Token bucket pacer for UDP sending
class UdpPacer {
 public:
  // rate_bps: target bitrate in bits per second
  // max_burst_bytes: maximum burst size in bytes
  UdpPacer(uint64_t rate_bps, std::size_t max_burst_bytes);

  // Consume tokens for sending bytes_to_send bytes
  // Returns the delay needed before sending
  std::chrono::microseconds ConsumeAndGetDelay(std::size_t bytes_to_send);

  // Update the target bitrate dynamically
  void UpdateRate(uint64_t new_rate_bps);

 private:
  uint64_t rate_bps_;
  std::size_t max_burst_bytes_;
  double tokens_;  // Current token count in bytes
  std::chrono::steady_clock::time_point last_update_;
};

}  // namespace mlvc::transport

#endif  // MLVC_TRANSPORT_UDP_PACER_H_
