#include <mlvc/core/status.h>
#include <mlvc/transport/udp_message_transport.h>

#include <cstdint>
#include <exception>
#include <iostream>
#include <vector>

int main() {
  try {
    constexpr uint16_t kPort = 45124;
    constexpr int kMessageCount = 900;
    mlvc::transport::UdpMessageReceiver receiver(kPort, "UDP burst test");
    mlvc::transport::UdpMessageSender sender("127.0.0.1", kPort, "UDP burst test");
    for (int index = 0; index < kMessageCount; ++index) {
      sender.Send({static_cast<uint8_t>(index & 0xff)});
      sender.Flush();
    }
    for (int index = 0; index < kMessageCount; ++index) {
      const std::vector<uint8_t> message = receiver.Receive();
      mlvc::Check(message.size() == 1 && message[0] == static_cast<uint8_t>(index & 0xff),
                  "UDP burst message order or content mismatch");
    }
    std::cout << "UDP receiver burst test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_udp_receiver_burst failed: " << error.what() << "\n";
    return 1;
  }
}
