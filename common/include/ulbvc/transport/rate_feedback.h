#pragma once

#include <chrono>
#include <cstdint>

#include "ulbvc/transport/rtcp_session.h"

namespace ulbvc::transport {
constexpr uint16_t kRateCapabilitiesTlv = 0x0100;
constexpr uint16_t kRateFeedbackTlv = 0x0101;
constexpr uint32_t kRateFeedbackVersion = 1;

struct RateCapabilities {
  uint64_t session_id = 0;
  uint32_t feedback_interval_ms = 500;
  uint32_t feedback_budget_bps = 25000;
};
struct RateFeedback {
  uint64_t session_id = 0;
  uint64_t receiver_instance = 0;
  uint32_t sequence = 0;
  uint32_t interval_ms = 0;
  uint64_t sample_time_ms = 0;  // Receiver-local uptime; only differences are compared.
  uint64_t received_wire_bytes = 0;
  uint64_t received_packets = 0;
  uint64_t missing_packets = 0;
  uint64_t recovered_packets = 0;
  uint32_t highest_sequence = 0;
  uint32_t jitter_ticks = 0;
  uint32_t queued_units = 0;
  uint32_t last_complete_frame = 0xffffffffu;
  uint32_t last_decoded_frame = 0xffffffffu;
};
struct RateFeedbackSnapshot {
  bool valid = false;
  RateFeedback feedback;
  std::chrono::steady_clock::time_point received_at{};
  uint32_t rtt_us = 0;
};

UlbvcControlTlv EncodeRateCapabilities(const RateCapabilities& value);
RateCapabilities DecodeRateCapabilities(const UlbvcControlTlv& value);
UlbvcControlTlv EncodeRateFeedback(const RateFeedback& value);
RateFeedback DecodeRateFeedback(const UlbvcControlTlv& value);
enum class RateControlMode : uint32_t { kFixedQ = 0, kTargetRate = 1, kAdaptiveRate = 2 };
struct RateCommand {
  std::optional<int> fixed_q, gop;
  std::optional<uint32_t> target_bitrate_bps;
  std::optional<RateControlMode> mode;
  bool random_access = false;
};
// Validates the entire transaction before any state is changed.
RateCommand DecodeRateCommand(const UlbvcControlMessage& message);
}  // namespace ulbvc::transport
