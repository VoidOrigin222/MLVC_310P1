#include "mlvc/transport/udp_pacer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace mlvc::transport {

UdpPacer::UdpPacer(uint64_t rate_bps, std::size_t max_burst_bytes)
    : rate_bps_(rate_bps),
      max_burst_bytes_(max_burst_bytes),
      tokens_(static_cast<double>(max_burst_bytes)),
      last_update_(std::chrono::steady_clock::now()) {}

std::chrono::microseconds UdpPacer::ConsumeAndGetDelay(std::size_t bytes_to_send) {
  if (rate_bps_ == 0) {
    return std::chrono::microseconds(0);
  }

  const auto now = std::chrono::steady_clock::now();
  const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - last_update_);

  // Add tokens based on elapsed time and rate
  const double tokens_to_add = (rate_bps_ / 8.0) * (elapsed.count() / 1e6);
  tokens_ = std::min(tokens_ + tokens_to_add, static_cast<double>(max_burst_bytes_));
  last_update_ = now;

  // Check if we have enough tokens
  if (tokens_ >= bytes_to_send) {
    tokens_ -= bytes_to_send;
    return std::chrono::microseconds(0);
  }

  // Calculate delay needed
  const double deficit = bytes_to_send - tokens_;
  const double delay_seconds = deficit / (rate_bps_ / 8.0);
  const auto delay_us = static_cast<int64_t>(delay_seconds * 1e6);

  tokens_ = 0;

  return std::chrono::microseconds(delay_us);
}

void UdpPacer::UpdateRate(uint64_t new_rate_bps) { rate_bps_ = new_rate_bps; }

}  // namespace mlvc::transport
