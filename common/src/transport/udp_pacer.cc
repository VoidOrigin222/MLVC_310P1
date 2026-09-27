#include "mlvc/transport/udp_pacer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace mlvc::transport {

UdpPacer::UdpPacer(uint64_t rate_bps, std::size_t max_burst_bytes)
    : rate_bps_(rate_bps),
      max_burst_bytes_(max_burst_bytes),
      tokens_(static_cast<double>(max_burst_bytes)),
      last_update_(std::chrono::steady_clock::now()) {
  if (max_burst_bytes_ == 0) throw std::invalid_argument("UDP pacer burst must be positive");
}

std::chrono::microseconds UdpPacer::ConsumeAndGetDelay(std::size_t bytes_to_send) {
  if (rate_bps_ == 0) {
    return std::chrono::microseconds(0);
  }
  if (bytes_to_send == 0) return std::chrono::microseconds(0);
  if (bytes_to_send > max_burst_bytes_) {
    throw std::invalid_argument("UDP packet exceeds the configured pacer burst");
  }

  const auto now = std::chrono::steady_clock::now();
  const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - last_update_);
  if (elapsed.count() > 0) {
    const long double tokens_to_add =
        (static_cast<long double>(rate_bps_) / 8.0L) * (elapsed.count() / 1e6L);
    if (!std::isfinite(tokens_to_add)) {
      throw std::overflow_error("UDP pacer token calculation overflow");
    }
    tokens_ = std::min(tokens_ + static_cast<double>(tokens_to_add),
                       static_cast<double>(max_burst_bytes_));
  }
  last_update_ = now;

  if (tokens_ >= static_cast<double>(bytes_to_send)) {
    tokens_ -= static_cast<double>(bytes_to_send);
    return std::chrono::microseconds(0);
  }

  // Keep an explicit negative token balance while the caller sleeps.  The
  // next call earns tokens against that debt, so elapsed wait time can never
  // be counted once for the current packet and again as free credit for the
  // following packet.
  const long double deficit = static_cast<long double>(bytes_to_send) - tokens_;
  const long double delay_value =
      std::ceil(deficit * 8000000.0L / static_cast<long double>(rate_bps_));
  if (!std::isfinite(delay_value) ||
      delay_value > static_cast<long double>(std::numeric_limits<int64_t>::max())) {
    throw std::overflow_error("UDP pacer delay exceeds the clock range");
  }
  const auto delay_us = static_cast<int64_t>(delay_value);
  tokens_ -= static_cast<double>(bytes_to_send);
  return std::chrono::microseconds(delay_us);
}

void UdpPacer::UpdateRate(uint64_t new_rate_bps) {
  rate_bps_ = new_rate_bps;
  last_update_ = std::chrono::steady_clock::now();
}

}  // namespace mlvc::transport
