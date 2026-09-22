#include <cassert>
#include <stdexcept>
#include <chrono>
#include <thread>

#include "mlvc/transport/rtp_mlvc.h"
using namespace mlvc::transport;
int main() {
  RtpMlvcReassembler r;
  std::vector<uint8_t> out;
  RtpMlvcFragment a{RtpUnitType::kEfu, 0, 1, 2, 0, 6, {1, 2, 3}},
      b{RtpUnitType::kEfu, 0, 1, 2, 3, 6, {4, 5, 6}};
  assert(!r.Push(b, &out));
  assert(r.Push(a, &out));
  assert((out == std::vector<uint8_t>{1, 2, 3, 4, 5, 6}));
  // A conflicting duplicate must be rejected while the unit is pending.
  RtpMlvcReassembler conflict;
  assert(!conflict.Push(a, &out));
  auto bad = a;
  bad.payload[0] = 9;
  bool threw = false;
  try {
    conflict.Push(bad, &out);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);
  // Reusing the same unit id after completion starts a fresh unit.
  assert(!r.Push(a, &out));
  RtpMlvcReassembler bounded(1024, 1, std::chrono::milliseconds(10));
  assert(!bounded.Push(a, &out));
  std::this_thread::sleep_for(std::chrono::milliseconds(15));
  assert(!bounded.Push(b, &out));
  auto rtcp = EncodeRtcpConfigRequest({7, 0x01020304});
  const auto decoded = DecodeRtcpConfigRequest(rtcp);
  assert(decoded.config_id == 7 && decoded.ssrc == 0x01020304);
  return 0;
}
