#include <cassert>
#include <stdexcept>

#include "mlvc/transport/rtp_mlvc.h"
using namespace mlvc::transport;
int main() {
  RtpMlvcFragment in{RtpUnitType::kEfu, kRtpMlvcStart | kRtpMlvcEnd, 0x01020304u, 0x11223344u, 0,
                     20, std::vector<uint8_t>(20, 0x44)};
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
  bad[0] = 2;
  expect_throw(bad);
  bad = bytes;
  bad.pop_back();
  expect_throw(bad);
  RtpMlvcFragment empty{RtpUnitType::kEos, kRtpMlvcStart | kRtpMlvcEnd, 7,
                        kMlvcNoReference, 0, 20,
                        std::vector<uint8_t>(20, 0x77)};
  assert(DecodeRtpMlvcFragment(EncodeRtpMlvcFragment(empty)).payload == empty.payload);

  // The RTP parser must skip negotiated CSRC and extension data and strip
  // valid RTP padding before handing the MLVC descriptor to the reassembler.
  const std::vector<uint8_t> descriptor = bytes;
  std::vector<uint8_t> rtp(12 + 4 + 4 + 4, 0);
  rtp[0] = 0xb1;  // version 2, padding, extension, one CSRC.
  rtp[1] = 0xe0;  // marker plus dynamic payload type 96.
  rtp[2] = 0x12;
  rtp[3] = 0x34;
  rtp[4] = 0x01;
  rtp[5] = 0x02;
  rtp[6] = 0x03;
  rtp[7] = 0x04;
  rtp[8] = 0xa0;
  rtp[9] = 0xa1;
  rtp[10] = 0xa2;
  rtp[11] = 0xa3;
  rtp[12] = 0xb0;
  rtp[13] = 0xb1;
  rtp[14] = 0xb2;
  rtp[15] = 0xb3;
  rtp[16] = 0x10;  // extension profile.
  rtp[17] = 0x00;
  rtp[18] = 0x00;  // one 32-bit extension word.
  rtp[19] = 0x01;
  rtp[20] = 0xc0;
  rtp[21] = 0xc1;
  rtp[22] = 0xc2;
  rtp[23] = 0xc3;
  rtp.insert(rtp.end(), descriptor.begin(), descriptor.end());
  rtp.push_back(0);
  rtp.push_back(2);
  const RtpPacket parsed = DecodeRtpPacket(rtp);
  assert(parsed.sequence == 0x1234 && parsed.timestamp == 0x01020304u);
  assert(parsed.ssrc == 0xa0a1a2a3u && parsed.marker);
  assert(parsed.payload == descriptor);

  // Receivers skip aligned descriptor extensions under X=1; extensions are
  // not exposed to the version-1 media-unit parser.
  std::vector<uint8_t> descriptor_extension(descriptor.begin(), descriptor.begin() + 24);
  descriptor_extension[2] |= kRtpMlvcExtension;
  descriptor_extension[4] = 0;
  descriptor_extension[5] = 28;
  descriptor_extension.insert(descriptor_extension.end(), {0xaa, 0xbb, 0xcc, 0xdd});
  descriptor_extension.insert(descriptor_extension.end(), descriptor.begin() + 24,
                              descriptor.end());
  const auto parsed_extension = DecodeRtpMlvcFragment(descriptor_extension);
  assert(parsed_extension.payload == in.payload && parsed_extension.flags == in.flags);

  RtpMlvcFragment optional_unit{static_cast<RtpUnitType>(0x05),
                                kRtpMlvcStart | kRtpMlvcEnd,
                                0,
                                42,
                                0,
                                20,
                                std::vector<uint8_t>(20, 0x33)};
  const auto parsed_optional = DecodeRtpMlvcFragment(EncodeRtpMlvcFragment(optional_unit));
  assert(static_cast<uint8_t>(parsed_optional.unit_type) == 0x05);

  auto reserved = bytes;
  reserved[3] = 1;
  expect_throw(reserved);
  auto unknown_type = bytes;
  unknown_type[1] = 0x7f;
  assert(static_cast<uint8_t>(DecodeRtpMlvcFragment(unknown_type).unit_type) == 0x7f);

  std::vector<uint8_t> optional_media_unit(20, 0);
  optional_media_unit[0] = kMlvcMediaUnitVersion;
  optional_media_unit[1] = 0x05;
  optional_media_unit[3] = 20;
  optional_media_unit[7] = 20;
  RtpMlvcFragment optional_rtp_unit{static_cast<RtpUnitType>(0x05),
                                    kRtpMlvcStart | kRtpMlvcEnd,
                                    0,
                                    0,
                                    0,
                                    20,
                                    optional_media_unit};
  RtpMlvcReassembler optional_reassembler;
  std::vector<uint8_t> optional_reassembled;
  assert(optional_reassembler.Push(
      DecodeRtpMlvcFragment(EncodeRtpMlvcFragment(optional_rtp_unit)),
      &optional_reassembled));
  assert(ParseMediaUnitHeader(optional_reassembled).unit_type == optional_rtp_unit.unit_type);
  ValidateMediaUnit(optional_reassembled);

  const auto expect_rtp_throw = [](const std::vector<uint8_t>& packet) {
    try {
      (void)DecodeRtpPacket(packet);
    } catch (const std::runtime_error&) {
      return;
    }
    assert(false);
  };
  auto malformed_rtp = rtp;
  malformed_rtp[0] = 0x81;  // One CSRC, but its bytes are absent.
  malformed_rtp.resize(12);
  expect_rtp_throw(malformed_rtp);
  malformed_rtp = rtp;
  malformed_rtp[19] = 0xff;  // Extension length exceeds packet.
  expect_rtp_throw(malformed_rtp);
  malformed_rtp = rtp;
  malformed_rtp.back() = 0;  // RTP padding length must be non-zero.
  expect_rtp_throw(malformed_rtp);
  malformed_rtp = rtp;
  malformed_rtp[1] = 0xdf;  // Static payload types are not valid MLVC PTs.
  expect_rtp_throw(malformed_rtp);
  bool static_payload_type_rejected = false;
  try {
    (void)DecodeRtpPacket(malformed_rtp, 95);
  } catch (const std::runtime_error&) {
    static_payload_type_rejected = true;
  }
  assert(static_payload_type_rejected);
  malformed_rtp = rtp;
  malformed_rtp[1] = 97;  // Not the negotiated dynamic payload type.
  expect_rtp_throw(malformed_rtp);
  return 0;
}
