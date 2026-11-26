#include <cassert>
#include <iostream>
#include <limits>

#include "ulbvc/transport/network_rate_controller.h"

using namespace ulbvc::transport;
namespace {
RateFeedbackSnapshot Sample(uint64_t now, uint32_t sequence, uint64_t packets, uint64_t bytes,
                            uint64_t lost = 0, uint64_t repaired = 0) {
  RateFeedbackSnapshot result;
  result.valid = true;
  result.received_at = std::chrono::steady_clock::time_point(std::chrono::milliseconds(now));
  result.rtt_us = 10000;
  result.feedback = {123,  456,      sequence, 500, now, bytes,      packets,
                     lost, repaired, 0,        0,   0,   0xffffffffu};
  return result;
}
template <class F>
void Reject(F f) {
  bool failed = false;
  try {
    f();
  } catch (const std::invalid_argument&) {
    failed = true;
  }
  assert(failed);
}
}  // namespace
int main() {
  auto feedback = Sample(500, 1, 100, 20000, 5, 3);
  const auto tlv = EncodeRateFeedback(feedback.feedback);
  assert(tlv.value.size() == 88);
  const auto decoded = DecodeRateFeedback(tlv);
  assert(decoded.session_id == 123 && decoded.receiver_instance == 456 &&
         decoded.sample_time_ms == 500);
  assert(decoded.received_wire_bytes == 20000 && decoded.missing_packets == 5 &&
         decoded.recovered_packets == 3);
  auto truncated = tlv;
  truncated.value.pop_back();
  Reject([&] { DecodeRateFeedback(truncated); });
  auto bad = tlv;
  bad.value[3] = 2;
  Reject([&] { DecodeRateFeedback(bad); });
  RateCapabilities caps{789, 500};
  assert(DecodeRateCapabilities(EncodeRateCapabilities(caps)).session_id == 789);
  caps.feedback_interval_ms = 0;
  Reject([&] { EncodeRateCapabilities(caps); });

  NetworkRateController quiet({2000000, 64000, 200, 3000});
  quiet.Update(0, {}, {});
  for (uint32_t i = 1; i <= 12; ++i)
    assert(quiet.Update(i * 500, {}, Sample(i * 500, i, i * 10, i * 1000)) == 2000000);
  assert(quiet.state() == NetworkRateState::kHealthy);  // App-limited traffic is not low capacity.
  auto duplicate = Sample(6500, 12, 120, 12000);
  quiet.Update(6500, {}, duplicate);
  assert(quiet.wire_bps() == 2000000);
  auto older = Sample(7000, 11, 130, 13000);
  quiet.Update(7000, {}, older);
  assert(quiet.wire_bps() == 2000000);
  // No auto-pause or accidental zero/unlimited target on feedback loss.
  for (uint64_t t = 10000; t <= 20000; t += 200) quiet.Update(t, {}, {});
  assert(quiet.wire_bps() == 64000 && quiet.state() == NetworkRateState::kSaturated);

  NetworkRateController busy({2000000, 64000, 200, 3000});
  busy.Update(0, {}, {});
  busy.Update(500, {}, Sample(500, 1, 100, 100000));
  busy.Update(1000, {}, Sample(1000, 2, 200, 200000, 30));
  assert(busy.wire_bps() == 2000000);
  busy.Update(1500, {}, Sample(1500, 3, 300, 300000, 60));
  assert(busy.wire_bps() == 1600000);
  const auto congested_rate = busy.wire_bps();
  // Feedback loss skips a sequence: use receiver uptime, not one report's window.
  busy.Update(2500, {}, Sample(2500, 5, 500, 500000, 60));
  assert(busy.received_bps() == 1600000);
  for (uint32_t i = 6; i <= 9; ++i)
    busy.Update(i * 500, {},
                Sample(i * 500, i, 500 + (i - 5) * 100, 500000 + (i - 5) * 100000, 60));
  assert(busy.wire_bps() >= congested_rate && busy.wire_bps() <= 2000000);
  auto restarted = Sample(5000, 1, 0, 0);
  restarted.feedback.receiver_instance = 789;
  busy.Update(5000, {}, restarted);
  assert(busy.state() == NetworkRateState::kStartup);
  auto slow = Sample(5500, 2, 100, 10000);
  slow.feedback.receiver_instance = 789;
  slow.feedback.queued_units = 12;
  const auto before = busy.wire_bps();
  busy.Update(5500, {}, slow);
  assert(busy.state() == NetworkRateState::kReceiverSlow && busy.wire_bps() == before);
  busy.Update(5700, {0, 100000, 200000}, {});
  assert(busy.wire_bps() == before && busy.state() == NetworkRateState::kEncoderLimited);
  Reject([&] { busy.SetMaximum(0); });

  UlbvcControlMessage command;
  command.type = UlbvcControlType::kCommand;
  command.flags = kUlbvcControlAtomic;
  command.transaction_id = 1;
  command.tlvs = {{1, {0, 0, 0, 11}}};
  assert(*DecodeRateCommand(command).fixed_q == 11);
  command.tlvs.push_back({99, {}});
  Reject([&] { DecodeRateCommand(command); });
  command.tlvs = {{1, {0, 0, 0, 64}}};
  Reject([&] { DecodeRateCommand(command); });
  command.tlvs = {{1, {0, 0, 0, 11}}, {5, {0, 0, 0, 2}}};
  Reject([&] { DecodeRateCommand(command); });
  command.tlvs = {{2, {0, 30, 132, 128}}, {5, {0, 0, 0, 1}}};
  assert(*DecodeRateCommand(command).target_bitrate_bps == 2000000);
  command.tlvs.push_back(command.tlvs[0]);
  Reject([&] { DecodeRateCommand(command); });
  std::cout << "network rate control: passed\n";
}
