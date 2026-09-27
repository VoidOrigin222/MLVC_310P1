#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "mlvc/transport/rtp_message_receiver.h"
#include "mlvc/transport/mlvc_media_unit.h"
#include "mlvc/transport/rtp_mlvc.h"

using mlvc::transport::kEfuRandomAccess;
using mlvc::transport::kEfuResetReference;
using mlvc::transport::kMlvcHeaderMaxBytes;
using mlvc::transport::kRtpMlvcEnd;
using mlvc::transport::kRtpMlvcStart;
using mlvc::transport::MlvcEfu;
using mlvc::transport::MlvcScu;
using mlvc::transport::ParseEfu;
using mlvc::transport::EncodeRtpMlvcFragment;
using mlvc::transport::RtpMessageReceiver;
using mlvc::transport::RtpMlvcFragment;
using mlvc::transport::RtpUnitType;
using mlvc::transport::SerializeEfu;
using mlvc::transport::SerializeScu;

namespace {

void Require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::vector<uint8_t> MakeRtpPacket(uint16_t sequence, uint32_t ssrc,
                                   const RtpMlvcFragment& fragment) {
  const std::vector<uint8_t> payload = EncodeRtpMlvcFragment(fragment);
  std::vector<uint8_t> packet(12, 0);
  packet[0] = 0x80;  // RTP version 2.
  const bool is_last_fragment =
      static_cast<uint64_t>(fragment.fragment_offset) + fragment.payload.size() ==
      fragment.unit_length;
  packet[1] = static_cast<uint8_t>(96 | (fragment.unit_type == RtpUnitType::kEfu &&
                                                is_last_fragment
                                            ? 0x80
                                            : 0));
  packet[2] = static_cast<uint8_t>(sequence >> 8);
  packet[3] = static_cast<uint8_t>(sequence);
  packet[4] = 0;
  packet[5] = 0;
  packet[6] = 0;
  packet[7] = 1;  // 90 kHz timestamp, only its presence matters here.
  packet[8] = static_cast<uint8_t>(ssrc >> 24);
  packet[9] = static_cast<uint8_t>(ssrc >> 16);
  packet[10] = static_cast<uint8_t>(ssrc >> 8);
  packet[11] = static_cast<uint8_t>(ssrc);
  packet.insert(packet.end(), payload.begin(), payload.end());
  return packet;
}

void SendPacket(int socket, uint16_t port, const std::vector<uint8_t>& packet) {
  sockaddr_in destination{};
  destination.sin_family = AF_INET;
  destination.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  destination.sin_port = htons(port);
  const ssize_t sent =
      ::sendto(socket, packet.data(), packet.size(), 0,
               reinterpret_cast<const sockaddr*>(&destination), sizeof(destination));
  Require(sent == static_cast<ssize_t>(packet.size()), "failed to send RTP test packet");
}

RtpMlvcFragment CompleteFragment(uint32_t unit_id, uint8_t value) {
  MlvcEfu efu;
  efu.config_id = 7;
  efu.frame_id = unit_id;
  efu.frame_type = 0;
  efu.unit_flags |= kEfuRandomAccess | kEfuResetReference;
  efu.entropy_payload = {value};
  const auto bytes = SerializeEfu(efu);
  return RtpMlvcFragment{RtpUnitType::kEfu, kRtpMlvcStart | kRtpMlvcEnd, 7, unit_id, 0,
                         static_cast<uint32_t>(bytes.size()), bytes};
}

uint8_t PayloadByte(const std::vector<uint8_t>& unit) {
  return ParseEfu(unit).entropy_payload.at(0);
}

int MakeSenderSocket() {
  const int socket = ::socket(AF_INET, SOCK_DGRAM, 0);
  Require(socket >= 0, "failed to create test sender socket");
  return socket;
}

void TestFirstAndContinuous() {
  RtpMessageReceiver receiver(0, "sequence test");
  const int sender = MakeSenderSocket();
  std::vector<uint8_t> unit;

  SendPacket(sender, receiver.local_port(), MakeRtpPacket(100, 0x11223344, CompleteFragment(1, 1)));
  Require(receiver.Receive(&unit), "first RTP packet did not complete a unit");
  Require(PayloadByte(unit) == 1, "first RTP unit mismatch");
  Require(receiver.lost_packets() == 0, "first RTP packet counted as lost");
  Require(receiver.duplicate_packets() == 0, "first RTP packet counted as duplicate");
  Require(receiver.reordered_packets() == 0, "first RTP packet counted as reordered");

  SendPacket(sender, receiver.local_port(), MakeRtpPacket(101, 0x11223344, CompleteFragment(2, 2)));
  Require(receiver.Receive(&unit), "continuous RTP packet did not complete a unit");
  Require(PayloadByte(unit) == 2, "continuous RTP unit mismatch");
  Require(receiver.lost_packets() == 0, "continuous RTP packet counted as lost");
  Require(receiver.duplicate_packets() == 0, "continuous RTP packet counted as duplicate");
  Require(receiver.reordered_packets() == 0, "continuous RTP packet counted as reordered");
  ::close(sender);
}

void TestLossAndDuplicate() {
  RtpMessageReceiver receiver(0, "sequence test");
  const int sender = MakeSenderSocket();
  std::vector<uint8_t> unit;

  SendPacket(sender, receiver.local_port(),
             MakeRtpPacket(10, 0x01020304, CompleteFragment(10, 10)));
  Require(receiver.Receive(&unit), "loss test first packet did not complete");
  SendPacket(sender, receiver.local_port(),
             MakeRtpPacket(13, 0x01020304, CompleteFragment(13, 13)));
  Require(receiver.Receive(&unit), "loss test second packet did not complete");
  Require(receiver.lost_packets() == 2, "RTP loss count mismatch");

  SendPacket(sender, receiver.local_port(),
             MakeRtpPacket(13, 0x01020304, CompleteFragment(13, 13)));
  SendPacket(sender, receiver.local_port(),
             MakeRtpPacket(14, 0x01020304, CompleteFragment(14, 14)));
  Require(receiver.Receive(&unit), "valid RTP packet after duplicate was not received");
  Require(PayloadByte(unit) == 14, "duplicate RTP packet was not discarded");
  Require(receiver.duplicate_packets() == 1, "RTP duplicate count mismatch");
  Require(receiver.lost_packets() == 2, "duplicate RTP packet changed the loss count");
  Require(receiver.reordered_packets() == 0, "duplicate RTP packet counted as reordered");
  ::close(sender);
}

void TestReorderStillReassembles() {
  RtpMessageReceiver receiver(0, "sequence test");
  const int sender = MakeSenderSocket();
  std::vector<uint8_t> unit;
  const auto complete = CompleteFragment(20, 0xaa);
  const std::size_t split = complete.payload.size() / 2;
  const RtpMlvcFragment tail{RtpUnitType::kEfu, kRtpMlvcEnd, 7, 20,
                             static_cast<uint32_t>(split),
                             static_cast<uint32_t>(complete.payload.size()),
                             std::vector<uint8_t>(complete.payload.begin() + split,
                                                  complete.payload.end())};
  const RtpMlvcFragment head{RtpUnitType::kEfu, kRtpMlvcStart, 7, 20, 0,
                             static_cast<uint32_t>(complete.payload.size()),
                             std::vector<uint8_t>(complete.payload.begin(),
                                                  complete.payload.begin() + split)};

  SendPacket(sender, receiver.local_port(), MakeRtpPacket(21, 0xaabbccdd, tail));
  SendPacket(sender, receiver.local_port(), MakeRtpPacket(20, 0xaabbccdd, head));
  Require(receiver.Receive(&unit), "reordered RTP head did not complete a unit");
  Require(PayloadByte(unit) == 0xaa, "reordered RTP unit mismatch");
  Require(receiver.reordered_packets() == 1, "RTP reorder count mismatch");
  Require(receiver.lost_packets() == 0, "reordered RTP packets counted as lost");
  ::close(sender);
}

void TestSsrcChangeRejected() {
  RtpMessageReceiver receiver(0, "sequence test");
  const int sender = MakeSenderSocket();
  std::vector<uint8_t> unit;

  SendPacket(sender, receiver.local_port(), MakeRtpPacket(1, 0x11111111, CompleteFragment(30, 30)));
  Require(receiver.Receive(&unit), "SSRC test first packet did not complete");
  SendPacket(sender, receiver.local_port(), MakeRtpPacket(2, 0x22222222, CompleteFragment(31, 31)));
  bool rejected = false;
  try {
    receiver.Receive(&unit);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  Require(rejected, "RTP SSRC change was not rejected");
  ::close(sender);
}

void TestLatePacketClosesLossGap() {
  RtpMessageReceiver receiver(0, "sequence test");
  const int sender = MakeSenderSocket();
  std::vector<uint8_t> unit;

  SendPacket(sender, receiver.local_port(), MakeRtpPacket(10, 0x55667788, CompleteFragment(10, 10)));
  Require(receiver.Receive(&unit), "first late-packet test unit did not complete");
  SendPacket(sender, receiver.local_port(), MakeRtpPacket(12, 0x55667788, CompleteFragment(12, 12)));
  Require(receiver.Receive(&unit), "unit after a sequence gap did not complete");
  Require(receiver.lost_packets() == 1, "sequence gap was not counted before late packet arrived");
  SendPacket(sender, receiver.local_port(), MakeRtpPacket(11, 0x55667788, CompleteFragment(11, 11)));
  Require(receiver.Receive(&unit), "late RTP unit did not complete");
  Require(PayloadByte(unit) == 11, "late RTP unit payload mismatch");
  Require(receiver.lost_packets() == 0, "late RTP packet did not close the loss gap");
  Require(receiver.reordered_packets() == 1, "late RTP packet was not counted as reordered");
  ::close(sender);
}

void TestMarkerMismatchRejected() {
  RtpMessageReceiver receiver(0, "marker test");
  const int sender = MakeSenderSocket();
  std::vector<uint8_t> unit;
  std::vector<uint8_t> packet =
      MakeRtpPacket(1, 0x12345678, CompleteFragment(40, 40));
  packet[1] = 96;  // Last EFU fragment must carry the RTP marker bit.
  SendPacket(sender, receiver.local_port(), packet);
  bool rejected = false;
  try {
    (void)receiver.Receive(&unit);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  Require(rejected, "RTP receiver accepted an EFU with a missing marker bit");
  ::close(sender);
}

void TestScuFrameLimitAppliedBeforeReassembly() {
  RtpMessageReceiver receiver(0, "SCU frame-limit test");
  const int sender = MakeSenderSocket();
  MlvcScu scu;
  scu.config_id = 9;
  scu.coded_width = scu.visible_width = 64;
  scu.coded_height = scu.visible_height = 64;
  scu.max_frame_bytes = 1;
  scu.codec_bundle_sha256.fill(0x42);
  const auto scu_bytes = SerializeScu(scu);
  const RtpMlvcFragment scu_fragment{RtpUnitType::kScu,
                                     kRtpMlvcStart | kRtpMlvcEnd,
                                     9,
                                     9,
                                     0,
                                     static_cast<uint32_t>(scu_bytes.size()),
                                     scu_bytes};
  SendPacket(sender, receiver.local_port(), MakeRtpPacket(1, 0xabcdef01, scu_fragment));
  std::vector<uint8_t> unit;
  Require(receiver.Receive(&unit), "SCU frame-limit setup was not received");

  const RtpMlvcFragment oversized_efu{RtpUnitType::kEfu,
                                      kRtpMlvcStart,
                                      9,
                                      0,
                                      0,
                                      static_cast<uint32_t>(kMlvcHeaderMaxBytes + 2),
                                      {0x01}};
  SendPacket(sender, receiver.local_port(),
             MakeRtpPacket(2, 0xabcdef01, oversized_efu));
  bool rejected = false;
  try {
    (void)receiver.Receive(&unit);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  Require(rejected, "RTP receiver allocated an EFU above its SCU frame-size limit");
  ::close(sender);
}

}  // namespace

int main() {
  try {
    TestFirstAndContinuous();
    TestLossAndDuplicate();
    TestReorderStillReassembles();
    TestLatePacketClosesLossGap();
    TestSsrcChangeRejected();
    TestMarkerMismatchRejected();
    TestScuFrameLimitAppliedBeforeReassembly();
    std::cout << "rtp message receiver sequence test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "rtp message receiver sequence test failed: " << error.what() << "\n";
    return 1;
  }
}
