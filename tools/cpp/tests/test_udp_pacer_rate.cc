#include "mlvc/transport/udp_pacer.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <thread>

int main() {
  try {
    using mlvc::transport::UdpPacer;
    UdpPacer pacer(80'000, 1'240);
    std::chrono::microseconds total{0};
    for (int index = 0; index < 21; ++index) {
      const auto delay = pacer.ConsumeAndGetDelay(1'000);
      total += delay;
      if (delay.count() > 0) std::this_thread::sleep_for(delay);
    }
    // One 1,240-byte burst is initially available; the remaining 20 packets
    // need about 2 seconds at 80 kbit/s.  We inspect the sender-thread delay
    // budget directly so this unit test does not sleep for wall-clock time.
    if (total < std::chrono::milliseconds(1'700) ||
        total > std::chrono::milliseconds(2'500)) {
      throw std::runtime_error("pacer delay budget is outside the configured rate");
    }

    bool rejected_oversized = false;
    try {
      (void)pacer.ConsumeAndGetDelay(1'241);
    } catch (const std::invalid_argument&) {
      rejected_oversized = true;
    }
    if (!rejected_oversized) {
      throw std::runtime_error("pacer accepted a packet larger than max_burst_bytes");
    }

    pacer.UpdateRate(0);
    if (pacer.ConsumeAndGetDelay(1'241).count() != 0) {
      throw std::runtime_error("disabled pacer did not bypass pacing");
    }
    std::cout << "udp pacer rate test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "udp pacer rate test failed: " << error.what() << "\n";
    return 1;
  }
}
