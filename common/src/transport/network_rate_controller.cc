#include "ulbvc/transport/network_rate_controller.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ulbvc::transport {
uint64_t MinimumAdaptiveWireRate(uint32_t ms) {
  if (!ms) throw std::invalid_argument("adaptive queue budget must be positive");
  const uint64_t denominator = uint64_t(ms) * 975;
  return std::max<uint64_t>(16000, (1240ull * 8000000 + denominator - 1) / denominator);
}
const char* NetworkRateStateName(NetworkRateState state) {
  switch (state) {
    case NetworkRateState::kStartup:
      return "startup";
    case NetworkRateState::kHealthy:
      return "healthy";
    case NetworkRateState::kCongested:
      return "congested";
    case NetworkRateState::kFeedbackLost:
      return "feedback_lost";
    case NetworkRateState::kSaturated:
      return "network_budget_saturated";
    case NetworkRateState::kReceiverSlow:
      return "receiver_slow";
    case NetworkRateState::kEncoderLimited:
      return "encoder_over_budget";
  }
  return "unknown";
}
NetworkRateController::NetworkRateController(NetworkRateOptions options) : options_(options) {
  if (options.max_wire_bps < 16000 || options.max_wire_bps > 100000000 || !options.queue_limit_ms ||
      options.queue_limit_ms > 60000 || options.feedback_timeout_ms < 1000 ||
      options.feedback_timeout_ms > 60000)
    throw std::invalid_argument("invalid adaptive rate options");
  // This is a control operating floor, not a claim about available bandwidth.
  const uint64_t packet_floor = MinimumAdaptiveWireRate(options.queue_limit_ms);
  minimum_ = options.min_wire_bps ? options.min_wire_bps : std::max<uint64_t>(16000, packet_floor);
  if (minimum_ > options.max_wire_bps || !minimum_)
    throw std::invalid_argument("invalid adaptive minimum");
  wire_bps_ = options.max_wire_bps;
}
void NetworkRateController::SetMaximum(uint64_t bps) {
  if (bps < minimum_ || bps > 100000000) throw std::invalid_argument("invalid adaptive maximum");
  options_.max_wire_bps = bps;
  wire_bps_ = std::min(wire_bps_, bps);
}
void NetworkRateController::Decrease() {
  wire_bps_ = std::max(minimum_, wire_bps_ * 4 / 5);
  healthy_windows_ = 0;
  state_ = wire_bps_ == minimum_ ? NetworkRateState::kSaturated : NetworkRateState::kCongested;
}
uint64_t NetworkRateController::Update(uint64_t now, const NetworkSendSample& sent,
                                       const RateFeedbackSnapshot& snapshot) {
  if (!started_) {
    started_ = true;
    last_tick_ = last_feedback_ms_ = now;
    last_send_bytes_ = sent.wire_bytes;
  }
  if (now < last_tick_) throw std::invalid_argument("network control clock moved backwards");
  const auto received_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(snapshot.received_at.time_since_epoch())
          .count();
  const bool valid = snapshot.valid && received_ms >= 0 && uint64_t(received_ms) <= now &&
                     now - uint64_t(received_ms) < options_.feedback_timeout_ms;
  const bool new_instance =
      valid &&
      (!have_feedback_ || snapshot.feedback.receiver_instance != prior_.receiver_instance ||
       snapshot.feedback.session_id != prior_.session_id);
  const bool fresh =
      valid &&
      (new_instance ||
       (static_cast<uint32_t>(snapshot.feedback.sequence - prior_.sequence) < 0x80000000u &&
        snapshot.feedback.sequence != prior_.sequence));
  if (fresh && new_instance) {
    prior_ = snapshot.feedback;
    have_feedback_ = true;
    last_feedback_ms_ = now;
    last_tick_ = now;
    last_send_bytes_ = sent.wire_bytes;
    bad_windows_ = healthy_windows_ = 0;
    baseline_rtt_us_ = snapshot.rtt_us;
    state_ = NetworkRateState::kStartup;
    return wire_bps_;
  }
  if (now - last_tick_ < 200) return wire_bps_;
  const bool queue_high =
      sent.oldest_queue_us > uint64_t(options_.queue_limit_ms) * 750 ||
      static_cast<long double>(sent.queued_bytes) * 8000 / std::max<uint64_t>(wire_bps_, 1) >
          options_.queue_limit_ms * 0.75L;
  if (fresh) {
    const auto& f = snapshot.feedback;
    if (f.received_wire_bytes < prior_.received_wire_bytes ||
        f.received_packets < prior_.received_packets ||
        f.missing_packets < prior_.missing_packets ||
        f.recovered_packets < prior_.recovered_packets || f.sample_time_ms <= prior_.sample_time_ms)
      return wire_bps_;
    if (f.sample_time_ms - prior_.sample_time_ms > 60000) {
      prior_ = f;
      last_feedback_ms_ = now;
      last_tick_ = now;
      last_send_bytes_ = sent.wire_bytes;
      bad_windows_ = healthy_windows_ = 0;
      state_ = NetworkRateState::kStartup;
      return wire_bps_;
    }
    const uint64_t packets = f.received_packets - prior_.received_packets;
    const uint64_t missing = f.missing_packets - prior_.missing_packets;
    const uint64_t repaired = f.recovered_packets - prior_.recovered_packets;
    const uint64_t lost = missing > repaired ? missing - repaired : 0;
    loss_fraction_ = static_cast<double>(lost) /
                     std::max<long double>(1, static_cast<long double>(packets) + lost);
    received_bps_ = static_cast<uint64_t>(std::min<long double>(
        1000000000, static_cast<long double>(f.received_wire_bytes - prior_.received_wire_bytes) *
                        8000 / (f.sample_time_ms - prior_.sample_time_ms)));
    if (snapshot.rtt_us && (!baseline_rtt_us_ || snapshot.rtt_us < baseline_rtt_us_))
      baseline_rtt_us_ = snapshot.rtt_us;
    const bool rtt_high = baseline_rtt_us_ && snapshot.rtt_us > baseline_rtt_us_ + 50000;
    const uint64_t elapsed = std::max<uint64_t>(1, now - last_feedback_ms_);
    const long double transmit_bps =
        sent.wire_bytes >= last_send_bytes_
            ? static_cast<long double>(sent.wire_bytes - last_send_bytes_) * 8000 / elapsed
            : 0;
    const bool bad = loss_fraction_ > 0.10 || (rtt_high && transmit_bps > wire_bps_ * 0.6L);
    bad_windows_ = bad ? bad_windows_ + 1 : 0;
    if (bad_windows_ >= 2)
      Decrease();
    else if ((f.queued_units > 8 ||
              (f.last_decoded_frame != 0xffffffffu && f.last_complete_frame != 0xffffffffu &&
               f.last_complete_frame > f.last_decoded_frame &&
               f.last_complete_frame - f.last_decoded_frame > 8)) &&
             !bad) {
      state_ = NetworkRateState::kReceiverSlow;
      healthy_windows_ = 0;
    } else if (!bad && loss_fraction_ < 0.01 && !rtt_high) {
      state_ = queue_high ? NetworkRateState::kEncoderLimited : NetworkRateState::kHealthy;
      // A quiet encoder cannot measure link capacity. Never lower its estimate
      // to observed throughput or generate filler to reach the configured target.
      if (++healthy_windows_ >= 4 && received_bps_ > wire_bps_ * 3 / 5) {
        wire_bps_ =
            std::min(options_.max_wire_bps, wire_bps_ + std::max<uint64_t>(1000, wire_bps_ / 20));
        healthy_windows_ = 0;
      }
    } else
      healthy_windows_ = 0;
    prior_ = f;
    last_feedback_ms_ = now;
    last_send_bytes_ = sent.wire_bytes;
  } else if (now - last_feedback_ms_ >= options_.feedback_timeout_ms) {
    // Missing feedback is not permission to pause video. Keep a positive bounded
    // operating rate and report the condition until the product policy is settled.
    Decrease();
    state_ = wire_bps_ == minimum_ ? NetworkRateState::kSaturated : NetworkRateState::kFeedbackLost;
  } else if (queue_high)
    state_ = NetworkRateState::kEncoderLimited;
  last_tick_ = now;
  return wire_bps_;
}
}  // namespace ulbvc::transport
