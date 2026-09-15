#include <cassert>
#include <stdexcept>

#include "mlvc/transport/rtp_mlvc.h"
using namespace mlvc::transport;
int main() {
  RtpMlvcFragment in{RtpUnitType::kEfu, 3, 0x01020304u, 0x11223344u, 5, 9, {1, 2, 3, 4}};
  auto bytes = EncodeRtpMlvcFragment(in);
  auto out = DecodeRtpMlvcFragment(bytes);
  assert(out.unit_type == in.unit_type && out.flags == in.flags);
  assert(out.config_id == in.config_id && out.unit_id == in.unit_id);
  assert(out.fragment_offset == in.fragment_offset && out.unit_length == in.unit_length);
  assert(out.payload == in.payload);
  auto expect_throw = [](std::vector<uint8_t> b) {
    try {
      DecodeRtpMlvcFragment(b);
    } catch (const std::runtime_error&) {
      return;
    }
    assert(false);
  };
  expect_throw({});
  auto bad = bytes;
  bad[0] = 'X';
  expect_throw(bad);
  bad = bytes;
  bad.pop_back();
  expect_throw(bad);
  RtpMlvcFragment empty{RtpUnitType::kEos, 0, 7, 8, 0, 0, {}};
  assert(DecodeRtpMlvcFragment(EncodeRtpMlvcFragment(empty)).payload.empty());
  return 0;
}
