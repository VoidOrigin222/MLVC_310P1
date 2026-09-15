#include <cassert>
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
  std::vector<uint8_t> output;
  bool complete = false;
  for (int i = 0; i < 8 && !complete; ++i) complete = receiver.Receive(&output);
  assert(complete);
  assert(output == input);
  assert(receiver.lost_packets() == 0);
  assert(receiver.duplicate_packets() == 0);
  return 0;
}
