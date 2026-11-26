#pragma once
#include <cstdint>

#include "ulbvc/transport/rate_feedback.h"

namespace ulbvc::transport {
uint64_t MinimumAdaptiveWireRate(uint32_t queue_limit_ms);
struct NetworkRateOptions {
  uint64_t max_wire_bps = 2000000;
  uint64_t min_wire_bps = 0;  // Derive the single-packet/queue budget floor when zero.
  uint32_t queue_limit_ms = 200;
  uint32_t feedback_timeout_ms = 3000;
};
struct NetworkSendSample {
  uint64_t wire_bytes = 0;  // All forward RTP + RTCP, including retransmission.
  uint64_t queued_bytes = 0;
  uint64_t oldest_queue_us = 0;
};
enum class NetworkRateState {
  kStartup,
  kHealthy,
  kCongested,
  kFeedbackLost,
  kSaturated,
  kReceiverSlow,
  kEncoderLimited
};
const char* NetworkRateStateName(NetworkRateState state);
class NetworkRateController {
 public:
  explicit NetworkRateController(NetworkRateOptions options);
  // Pure control logic with caller-supplied monotonic milliseconds; safe to simulate.
  uint64_t Update(uint64_t now_ms, const NetworkSendSample& sent,
                  const RateFeedbackSnapshot& feedback);
  uint64_t wire_bps() const { return wire_bps_; }
  NetworkRateState state() const { return state_; }
  double loss_fraction() const { return loss_fraction_; }
  uint64_t received_bps() const { return received_bps_; }
  void SetMaximum(uint64_t bps);

 private:
  void Decrease();
  NetworkRateOptions options_;
  uint64_t wire_bps_ = 0, minimum_ = 0;
  uint64_t last_tick_ = 0, last_feedback_ms_ = 0, last_send_bytes_ = 0;
  bool started_ = false, have_feedback_ = false;
  RateFeedback prior_;
  unsigned bad_windows_ = 0, healthy_windows_ = 0;
  uint32_t baseline_rtt_us_ = 0;
  double loss_fraction_ = 0;
  uint64_t received_bps_ = 0;
  NetworkRateState state_ = NetworkRateState::kStartup;
};
}  // namespace ulbvc::transport
