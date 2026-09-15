#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "mlvc/transport/rtp_message_receiver.h"

using mlvc::transport::EncodeRtpMlvcFragment;
using mlvc::transport::RtpMessageReceiver;
using mlvc::transport::RtpMlvcFragment;
using mlvc::transport::RtpUnitType;

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
  packet[1] = 96;    // Dynamic payload type used by MLVC RTP.
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
  return RtpMlvcFragment{RtpUnitType::kEfu, 0, 7, unit_id, 0, 1, {value}};
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
  Require(unit == std::vector<uint8_t>{1}, "first RTP unit mismatch");
  Require(receiver.lost_packets() == 0, "first RTP packet counted as lost");
  Require(receiver.duplicate_packets() == 0, "first RTP packet counted as duplicate");
  Require(receiver.reordered_packets() == 0, "first RTP packet counted as reordered");

  SendPacket(sender, receiver.local_port(), MakeRtpPacket(101, 0x11223344, CompleteFragment(2, 2)));
  Require(receiver.Receive(&unit), "continuous RTP packet did not complete a unit");
  Require(unit == std::vector<uint8_t>{2}, "continuous RTP unit mismatch");
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
  Require(receiver.Receive(&unit), "duplicate RTP packet did not reach reassembler");
  Require(receiver.duplicate_packets() == 1, "RTP duplicate count mismatch");
  Require(receiver.reordered_packets() == 0, "duplicate RTP packet counted as reordered");
  ::close(sender);
}

void TestReorderStillReassembles() {
  RtpMessageReceiver receiver(0, "sequence test");
  const int sender = MakeSenderSocket();
  std::vector<uint8_t> unit;
  const RtpMlvcFragment tail{RtpUnitType::kEfu, 0, 9, 20, 1, 2, {0xbb}};
  const RtpMlvcFragment head{RtpUnitType::kEfu, 0, 9, 20, 0, 2, {0xaa}};

  SendPacket(sender, receiver.local_port(), MakeRtpPacket(21, 0xaabbccdd, tail));
  SendPacket(sender, receiver.local_port(), MakeRtpPacket(20, 0xaabbccdd, head));
  Require(receiver.Receive(&unit), "reordered RTP head did not complete a unit");
  Require(unit == std::vector<uint8_t>({0xaa, 0xbb}), "reordered RTP unit mismatch");
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

}  // namespace

int main() {
  try {
    TestFirstAndContinuous();
    TestLossAndDuplicate();
    TestReorderStillReassembles();
    TestSsrcChangeRejected();
    std::cout << "rtp message receiver sequence test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "rtp message receiver sequence test failed: " << error.what() << "\n";
    return 1;
  }
}
