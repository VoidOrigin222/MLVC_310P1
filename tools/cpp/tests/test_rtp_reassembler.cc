#include <cassert>
#include <stdexcept>

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
  return 0;
}
