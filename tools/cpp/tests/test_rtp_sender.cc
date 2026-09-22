#include <cassert>
#include <chrono>
#include <cstdint>
#include <vector>

#include "mlvc/transport/rtp_message_receiver.h"
#include "mlvc/transport/rtp_mlvc.h"

int main() {
  mlvc::transport::RtpMessageReceiver receiver(0, "sender test receiver");
  mlvc::transport::RtpMlvcSender sender("127.0.0.1", receiver.local_port(), 0x01020304u);
  std::vector<uint8_t> input(5000);
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = static_cast<uint8_t>(i * 31u);
  sender.SendUnit(mlvc::transport::RtpUnitType::kEfu, 0, 7, 9, 9000, input);
  sender.Flush();
  const auto stats = sender.Stats();
  assert(stats.packets == 5);
  assert(stats.wire_bytes > input.size());
  std::vector<uint8_t> output;
  bool complete = false;
  for (int i = 0; i < 8 && !complete; ++i) complete = receiver.Receive(&output);
  assert(complete);
  assert(output == input);
  assert(receiver.lost_packets() == 0);
  assert(receiver.duplicate_packets() == 0);

  mlvc::transport::RtpMlvcSender random_sender("127.0.0.1", receiver.local_port());
  random_sender.SetSessionConfig(42, {1, 2, 3});
  random_sender.ResendSessionConfig(9001);
  random_sender.Flush();

  // Pacing belongs to the sender thread: enqueueing a unit must not sleep on
  // the encoder thread, while Flush waits for paced transmission to finish.
  mlvc::transport::RtpMlvcSender paced("127.0.0.1", receiver.local_port(), 0x01020305u,
                                       80000, 1200, 64 * 1024, 2000);
  const auto enqueue_begin = std::chrono::steady_clock::now();
  paced.SendUnit(mlvc::transport::RtpUnitType::kEfu, 0, 8, 10, 12000, input);
  const auto enqueue_elapsed = std::chrono::steady_clock::now() - enqueue_begin;
  assert(enqueue_elapsed < std::chrono::milliseconds(100));
  const auto flush_begin = std::chrono::steady_clock::now();
  paced.Flush();
  const auto flush_elapsed = std::chrono::steady_clock::now() - flush_begin;
  assert(flush_elapsed >= std::chrono::milliseconds(200));
  assert(paced.Stats().packets == 5);
  return 0;
}
