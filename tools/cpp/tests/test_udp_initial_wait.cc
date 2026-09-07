#include <mlvc/core/status.h>
#include <mlvc/transport/udp_message_transport.h>

#include <chrono>
#include <exception>
#include <future>
#include <iostream>
#include <thread>
#include <unistd.h>
#include <vector>

int main() {
  try {
    constexpr int kLegacyInitialWaitSeconds = 70;
    const uint16_t port = static_cast<uint16_t>(40000 + (getpid() % 20000));
    const std::vector<uint8_t> expected{0x01, 0x02, 0x03, 0x04};
    mlvc::transport::UdpMessageReceiver receiver(port, "initial-wait test");
    std::future<std::vector<uint8_t>> received =
        std::async(std::launch::async, [&receiver] { return receiver.Receive(); });

    std::this_thread::sleep_for(std::chrono::seconds(kLegacyInitialWaitSeconds));
    mlvc::transport::UdpMessageSender sender("127.0.0.1", port, "initial-wait test");
    sender.Send(expected);
    sender.Flush();

    mlvc::Check(received.get() == expected,
                "receiver did not accept a first message after the legacy timeout window");
    std::cout << "udp initial wait test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "udp initial wait test failed: " << error.what() << "\n";
    return 1;
  }
}
