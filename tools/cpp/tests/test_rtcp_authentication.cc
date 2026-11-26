#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cassert>
#include <cstdlib>
#include <iostream>

#include "ulbvc/transport/rtcp_session.h"

using namespace ulbvc::transport;
int main() {
  setenv("ULBVC_RTCP_KEY", "local-unit-test-key", 1);
  const int tap = ::socket(AF_INET, SOCK_DGRAM, 0);
  assert(tap >= 0);
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  assert(::bind(tap, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0);
  socklen_t length = sizeof(address);
  assert(::getsockname(tap, reinterpret_cast<sockaddr*>(&address), &length) == 0);
  timeval timeout{2, 0};
  assert(::setsockopt(tap, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0);
  RtcpUdpEndpoint receiver(0);
  RtcpUdpEndpoint sender(0, "127.0.0.1", ntohs(address.sin_port));
  sockaddr_in destination = address;
  destination.sin_port = htons(receiver.local_port());
  const auto capture = [&](RtcpUdpEndpoint& from, uint32_t ssrc) {
    RtcpReceiverReport rr;
    rr.sender_ssrc = ssrc;
    from.Send(EncodeRtcpReceiverReport(rr, "auth-unit-test"));
    std::vector<uint8_t> wire(4096);
    const auto bytes = ::recv(tap, wire.data(), wire.size(), 0);
    assert(bytes > 0);
    wire.resize(bytes);
    return wire;
  };
  const auto deliver = [&](const std::vector<uint8_t>& wire) {
    assert(::sendto(tap, wire.data(), wire.size(), 0, reinterpret_cast<sockaddr*>(&destination),
                    sizeof(destination)) == ssize_t(wire.size()));
    std::vector<uint8_t> decoded;
    return receiver.Receive(&decoded, std::chrono::milliseconds(100));
  };
  const auto first = capture(sender, 0x12345678);
  assert(deliver(first));
  assert(!deliver(first));
  auto tampered = capture(sender, 0x12345678);
  tampered.back() ^= 1;
  assert(!deliver(tampered));
  RtcpUdpEndpoint restarted(0, "127.0.0.1", ntohs(address.sin_port));
  assert(deliver(capture(restarted, 0x87654321)));  // Fresh SSRC, index starts at zero.
  assert(!deliver(first));
  setenv("ULBVC_RTCP_KEY", "wrong-local-unit-test-key", 1);
  RtcpUdpEndpoint wrong(0, "127.0.0.1", ntohs(address.sin_port));
  assert(!deliver(capture(wrong, 0xabcdef01)));
  unsetenv("ULBVC_RTCP_KEY");
  ::close(tap);
  std::cout << "RTCP authentication/replay/restart: passed\n";
}
