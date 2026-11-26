#include "ulbvc/transport/rate_feedback.h"

#include <limits>
#include <set>
#include <stdexcept>

namespace ulbvc::transport {
namespace {
void Put(std::vector<uint8_t>* out, uint64_t value, unsigned bytes) {
  for (unsigned i = bytes; i; --i) out->push_back(static_cast<uint8_t>(value >> ((i - 1) * 8)));
}
uint64_t Get(const std::vector<uint8_t>& data, std::size_t* at, unsigned bytes) {
  if (*at > data.size() || bytes > data.size() - *at)
    throw std::invalid_argument("truncated rate feedback");
  uint64_t value = 0;
  while (bytes--) value = (value << 8) | data[(*at)++];
  return value;
}
void Check(bool ok) {
  if (!ok) throw std::invalid_argument("invalid rate feedback field or version");
}
}  // namespace
UlbvcControlTlv EncodeRateCapabilities(const RateCapabilities& value) {
  Check(value.session_id && value.feedback_interval_ms >= 100 &&
        value.feedback_interval_ms <= 5000 && value.feedback_budget_bps >= 2048 &&
        value.feedback_budget_bps <= 2500000);
  UlbvcControlTlv out{kRateCapabilitiesTlv, {}};
  Put(&out.value, kRateFeedbackVersion, 4);
  Put(&out.value, value.session_id, 8);
  Put(&out.value, value.feedback_interval_ms, 4);
  Put(&out.value, value.feedback_budget_bps, 4);
  return out;
}
RateCapabilities DecodeRateCapabilities(const UlbvcControlTlv& value) {
  Check(value.type == kRateCapabilitiesTlv && value.value.size() == 20);
  std::size_t at = 0;
  Check(Get(value.value, &at, 4) == kRateFeedbackVersion);
  RateCapabilities out;
  out.session_id = Get(value.value, &at, 8);
  out.feedback_interval_ms = static_cast<uint32_t>(Get(value.value, &at, 4));
  out.feedback_budget_bps = static_cast<uint32_t>(Get(value.value, &at, 4));
  (void)EncodeRateCapabilities(out);
  return out;
}
UlbvcControlTlv EncodeRateFeedback(const RateFeedback& v) {
  Check(v.session_id && v.receiver_instance && v.interval_ms > 0 && v.interval_ms <= 60000 &&
        v.recovered_packets <= v.missing_packets && v.queued_units <= 64);
  UlbvcControlTlv out{kRateFeedbackTlv, {}};
  Put(&out.value, kRateFeedbackVersion, 4);
  Put(&out.value, v.session_id, 8);
  Put(&out.value, v.receiver_instance, 8);
  Put(&out.value, v.sequence, 4);
  Put(&out.value, v.interval_ms, 4);
  Put(&out.value, v.sample_time_ms, 8);
  Put(&out.value, v.received_wire_bytes, 8);
  Put(&out.value, v.received_packets, 8);
  Put(&out.value, v.missing_packets, 8);
  Put(&out.value, v.recovered_packets, 8);
  Put(&out.value, v.highest_sequence, 4);
  Put(&out.value, v.jitter_ticks, 4);
  Put(&out.value, v.queued_units, 4);
  Put(&out.value, v.last_complete_frame, 4);
  Put(&out.value, v.last_decoded_frame, 4);
  return out;
}
RateFeedback DecodeRateFeedback(const UlbvcControlTlv& value) {
  Check(value.type == kRateFeedbackTlv && value.value.size() == 88);
  std::size_t at = 0;
  Check(Get(value.value, &at, 4) == kRateFeedbackVersion);
  RateFeedback v;
  v.session_id = Get(value.value, &at, 8);
  v.receiver_instance = Get(value.value, &at, 8);
  v.sequence = static_cast<uint32_t>(Get(value.value, &at, 4));
  v.interval_ms = static_cast<uint32_t>(Get(value.value, &at, 4));
  v.sample_time_ms = Get(value.value, &at, 8);
  v.received_wire_bytes = Get(value.value, &at, 8);
  v.received_packets = Get(value.value, &at, 8);
  v.missing_packets = Get(value.value, &at, 8);
  v.recovered_packets = Get(value.value, &at, 8);
  v.highest_sequence = static_cast<uint32_t>(Get(value.value, &at, 4));
  v.jitter_ticks = static_cast<uint32_t>(Get(value.value, &at, 4));
  v.queued_units = static_cast<uint32_t>(Get(value.value, &at, 4));
  v.last_complete_frame = static_cast<uint32_t>(Get(value.value, &at, 4));
  v.last_decoded_frame = static_cast<uint32_t>(Get(value.value, &at, 4));
  (void)EncodeRateFeedback(v);
  return v;
}
RateCommand DecodeRateCommand(const UlbvcControlMessage& message) {
  Check(message.version == 1 && message.type == UlbvcControlType::kCommand &&
        message.flags == kUlbvcControlAtomic && message.transaction_id && !message.tlvs.empty() &&
        (message.apply_after_frame_id == 0xffffffffu ||
         message.apply_after_frame_id <= uint32_t(std::numeric_limits<int>::max())));
  RateCommand result;
  std::set<uint16_t> seen;
  for (const auto& tlv : message.tlvs) {
    Check(seen.insert(tlv.type).second);
    if (tlv.type == 4) {
      Check(tlv.value.empty() ||
            (tlv.value.size() == 4 && tlv.value == std::vector<uint8_t>{0, 0, 0, 1}));
      result.random_access = true;
      continue;
    }
    Check(tlv.value.size() == 4);
    std::size_t at = 0;
    const auto value = static_cast<uint32_t>(Get(tlv.value, &at, 4));
    switch (tlv.type) {
      case 1:
        Check(value <= 63);
        result.fixed_q = int(value);
        break;
      case 2:
        Check(value >= 1 && value <= 100000000);
        result.target_bitrate_bps = value;
        break;
      case 3:
        Check(value >= 1 && value <= 1000000);
        result.gop = int(value);
        break;
      case 5:
        Check(value <= 2);
        result.mode = static_cast<RateControlMode>(value);
        break;
      default:
        throw std::invalid_argument("unsupported rate control TLV");
    }
  }
  Check(!(result.fixed_q && result.target_bitrate_bps));
  if (result.fixed_q) Check(!result.mode || *result.mode == RateControlMode::kFixedQ);
  if (result.target_bitrate_bps)
    Check(!result.mode || *result.mode == RateControlMode::kTargetRate);
  return result;
}
}  // namespace ulbvc::transport
