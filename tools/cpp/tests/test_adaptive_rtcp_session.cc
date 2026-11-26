#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>

#include "ulbvc/transport/rtp_message_receiver.h"

using namespace ulbvc::transport;
namespace {
template <class Predicate>
void Wait(Predicate predicate) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(4);
  while (!predicate() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  assert(predicate());
}
}  // namespace
int main() {
  auto receiver = std::make_unique<RtpMessageReceiver>(0, "adaptive loopback");
  const auto port = receiver->local_port();
  RtpUlbvcSender sender("127.0.0.1", port, 0x12345678, 2000000, 1240, 1024 * 1024, 200, 96,
                        std::chrono::milliseconds(100));
  sender.EnableRateFeedback(100);
  UlbvcScu scu;
  scu.config_id = 1;
  scu.coded_width = scu.visible_width = 64;
  scu.coded_height = scu.visible_height = 64;
  scu.max_frame_bytes = 200000;
  scu.codec_bundle_sha256.fill(0x3a);
  const auto config = SerializeScu(scu);
  sender.SetSessionConfig(1, config);
  UlbvcEfu efu;
  efu.config_id = 1;
  efu.frame_id = 0;
  efu.frame_type = 0;
  efu.unit_flags |= kEfuRandomAccess | kEfuResetReference;
  efu.entropy_payload.resize(20000, 0x42);
  const auto frame = SerializeEfu(efu);
  sender.SendUnit(RtpUnitType::kEfu, 0, 1, 0, 9000, frame);
  sender.Flush();
  std::vector<uint8_t> bytes;
  assert(receiver->ReceiveFor(&bytes, std::chrono::seconds(2)) && bytes == config);
  assert(receiver->ReceiveFor(&bytes, std::chrono::seconds(2)) && bytes == frame);
  receiver->ReportDecodedFrame(0);
  Wait([&] {
    return sender.Feedback().valid && sender.Feedback().feedback.last_decoded_frame == 0;
  });
  const auto first = sender.Feedback();
  assert(first.feedback.received_packets > 1 &&
         first.feedback.received_wire_bytes == sender.Stats().wire_bytes);
  assert(sender.Stats().rtcp_wire_bytes > 0);

  receiver->RequestRandomAccess();
  bool requested = false;
  Wait([&] {
    requested = requested || sender.ConsumeRandomAccessRequest();
    return requested;
  });

  // Mid-frame rate changes must wake blocked queue admission and pacing waits.
  efu.frame_id = 1;
  efu.entropy_payload.resize(80000, 0x43);
  const auto large = SerializeEfu(efu);
  std::thread producer([&] { sender.SendUnit(RtpUnitType::kEfu, 0, 1, 1, 12000, large); });
  Wait([&] { return sender.Stats().queued_wire_bytes > 1000; });
  sender.SetPacingRate(200000);
  assert(receiver->ReceiveFor(&bytes, std::chrono::seconds(6)) && bytes == large);
  producer.join();
  sender.Flush();
  assert(sender.Stats().queued_wire_bytes == 0 && sender.Stats().pacing_rate_bps == 200000);
  bool rejected = false;
  try {
    sender.SetPacingRate(0);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);

  // Receiver restart: new control SSRC/instance must be accepted without an encoder restart.
  const auto old_instance = sender.Feedback().feedback.receiver_instance;
  receiver.reset();
  receiver = std::make_unique<RtpMessageReceiver>(port, "adaptive restarted receiver");
  sender.ResendSessionConfig(15000);
  assert(receiver->ReceiveFor(&bytes, std::chrono::seconds(2)) && bytes == config);
  Wait([&] {
    return sender.Feedback().valid && sender.Feedback().feedback.receiver_instance != old_instance;
  });
  assert(sender.Feedback().feedback.session_id == first.feedback.session_id);
  std::cout << "adaptive RTP/RTCP loopback: passed\n";
}
