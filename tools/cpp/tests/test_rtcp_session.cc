#include "mlvc/transport/rtcp_session.h"

#include <cassert>
#include <iostream>
#include <stdexcept>

int main() {
  try {
    const auto packet = mlvc::transport::EncodeRtcpSessionBye(0x01020304u, "mlvc-test");
    assert(mlvc::transport::DecodeRtcpSessionBye(packet) == 0x01020304u);
    auto malformed = packet;
    malformed[3] = 0;
    bool rejected = false;
    try {
      (void)mlvc::transport::DecodeRtcpSessionBye(malformed);
    } catch (const std::exception&) {
      rejected = true;
    }
    assert(rejected);
    std::cout << "rtcp session test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "rtcp session test failed: " << error.what() << "\n";
    return 1;
  }
}
