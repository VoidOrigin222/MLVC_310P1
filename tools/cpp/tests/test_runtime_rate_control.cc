#include <algorithm>
#include <cassert>
#include <iostream>

#include "ulbvc/codec/ulbvc_rate_control.h"
#include "ulbvc/transport/network_rate_controller.h"

using namespace ulbvc::codec;
namespace {
// Virtual saturated-link feedback: 2 Mb/s -> 500 kb/s -> 2 Mb/s.
// Exercise the network and codec controllers together without a board or sleeps.
void TestBandwidthSteps() {
  using namespace ulbvc::transport;
  NetworkRateController network({2000000, 64000, 200, 3000});
  UlbvcRateController codec({1920, 1080, 30, 1600000, 11, 0, 11});
  uint64_t sent_bytes = 0, received_bytes = 0, packets = 0, missing = 0;
  int frame = 0, minimum_low_q = 63, last_recovered_q = 0;
  uint64_t minimum_budget = 2000000;
  for (uint32_t window = 1; window <= 140; ++window) {
    const uint64_t capacity = window > 10 && window <= 50 ? 500000 : 2000000;
    const uint64_t offered = network.wire_bps() * 975 / 1000;
    const uint64_t delivered = std::min(capacity, offered);
    sent_bytes += offered / 16;
    received_bytes += delivered / 16;
    packets += std::max<uint64_t>(1, delivered / 16000);
    missing += (offered - delivered) / 16000;
    RateFeedbackSnapshot feedback;
    feedback.valid = true;
    feedback.received_at =
        std::chrono::steady_clock::time_point(std::chrono::milliseconds(window * 500));
    feedback.rtt_us = 10000;
    feedback.feedback = {123,     456,     window, 500, window * 500ull, received_bytes,
                         packets, missing, 0};
    const auto budget = network.Update(window * 500, {sent_bytes, 0, 0}, feedback);
    codec.SetTargetBitrate(budget * 0.8);
    if (window > 10 && window <= 50) minimum_budget = std::min(minimum_budget, budget);
    for (int index = 0; index < 15; ++index, ++frame) {
      const auto pts = frame / 30.0;
      const auto type = frame % 96 == 0 ? UlbvcFrameType::kIFrame : UlbvcFrameType::kPFrame;
      const int q = codec.SolveQIndex(pts, type);
      // Deliberately use measured-size observations with scene-model mismatch.
      codec.Update(pts, type, q, static_cast<std::size_t>(1000 + 350 * q));
      if (window > 20 && window <= 50) minimum_low_q = std::min(minimum_low_q, q);
      if (window > 130) last_recovered_q = std::max(last_recovered_q, q);
    }
  }
  assert(minimum_budget < 700000 && minimum_low_q < 11);
  assert(network.wire_bps() > minimum_budget && network.wire_bps() <= 2000000);
  assert(last_recovered_q > minimum_low_q);
}
}  // namespace
int main() {
  TestBandwidthSteps();
  UlbvcRateController rate({1920, 1080, 30, 2000000, 11, 0, 11});
  const auto i = UlbvcFrameType::kIFrame, p = UlbvcFrameType::kPFrame;
  const int q0 = rate.SolveQIndex(0, i);
  const int q1 = rate.SolveQIndex(1.0 / 30, p);
  const int q2 = rate.SolveQIndex(2.0 / 30, p);
  rate.SetTargetBitrate(500000);  // Old reservations must remain valid.
  rate.Update(0, i, q0, 20000);
  const int q3 = rate.SolveQIndex(3.0 / 30, p);
  rate.Update(1.0 / 30, p, q1, 3000);
  rate.Update(2.0 / 30, p, q2, 3000);
  rate.Update(3.0 / 30, p, q3, 3000);
  assert(rate.target_bitrate_bps() == 500000);
  int smallest = 63;
  for (int f = 4; f < 100; ++f) {
    const auto pts = f / 30.0;
    const int q = rate.SolveQIndex(pts, p);
    smallest = std::min(smallest, q);
    rate.Update(pts, p, q, 10000);
    assert(q >= 0 && q <= 11);
  }
  assert(smallest < 11);
  const int old_q = rate.SolveQIndex(100.0 / 30, p);
  rate.SetTargetBitrate(0);
  assert(!rate.enabled());
  const int fixed = rate.SolveQIndex(101.0 / 30, p);
  assert(fixed == 11);
  rate.SetTargetBitrate(1000000);
  rate.Update(100.0 / 30, p, old_q, 1000);
  rate.Update(101.0 / 30, p, fixed, 1000);
  bool bad = false;
  try {
    rate.SetTargetBitrate(-1);
  } catch (const std::exception&) {
    bad = true;
  }
  assert(bad);
  bad = false;
  try {
    rate.SolveQIndex(101.0 / 30, p);
  } catch (const std::exception&) {
    bad = true;
  }
  assert(bad);
  std::cout << "runtime rate control: passed\n";
}
