#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cassert>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

#include "mlvc/io/udp_frame_transport.h"

namespace {

using mlvc::codec::MlvcFrameType;
using mlvc::transport::MlvcEfu;
using mlvc::transport::MlvcScu;
using mlvc::transport::RtpMlvcFragment;
using mlvc::transport::RtpUnitType;

std::vector<uint8_t> MakeScu(uint32_t config_id) {
  MlvcScu scu;
  scu.config_id = config_id;
  scu.coded_width = scu.visible_width = 1920;
  scu.coded_height = 1088;
  scu.visible_height = 1080;
  for (std::size_t i = 0; i < scu.codec_bundle_sha256.size(); ++i) {
    scu.codec_bundle_sha256[i] = static_cast<uint8_t>(0x30 + i);
  }
  return mlvc::transport::SerializeScu(scu);
}

std::vector<uint8_t> MakeEfu(uint32_t config_id, uint32_t frame_id, MlvcFrameType type,
                             uint32_t short_ref, uint32_t long_ref, bool store_ltr = false) {
  MlvcEfu efu;
  efu.config_id = config_id;
  efu.frame_id = frame_id;
  efu.frame_type = static_cast<uint8_t>(type);
  efu.entropy_q_index = 10;
  efu.model_q_index = 10;
  efu.short_ref_frame_id = short_ref;
  efu.long_ref_frame_id = long_ref;
  if (type == MlvcFrameType::kIFrame) {
    efu.unit_flags |= mlvc::transport::kEfuRandomAccess |
                      mlvc::transport::kEfuResetReference;
  }
  if (type == MlvcFrameType::kLtrRecovery) {
    efu.unit_flags |= mlvc::transport::kEfuResetReference;
  }
  if (store_ltr) efu.unit_flags |= mlvc::transport::kEfuStoreAsLtr;
  efu.entropy_payload = {static_cast<uint8_t>(frame_id)};
  return mlvc::transport::SerializeEfu(efu);
}

void SendUnit(int socket, uint16_t port, uint16_t* sequence, uint32_t timestamp,
              const std::vector<uint8_t>& unit) {
  const auto header = mlvc::transport::ParseMediaUnitHeader(unit);
  const RtpUnitType type = header.unit_type;
  const bool is_efu = type == RtpUnitType::kEfu;
  const uint8_t flags = mlvc::transport::kRtpMlvcStart | mlvc::transport::kRtpMlvcEnd;
  const RtpMlvcFragment fragment{type, flags, header.config_id, header.unit_id, 0,
                                 header.unit_length, unit};
  const auto payload = mlvc::transport::EncodeRtpMlvcFragment(fragment);
  std::vector<uint8_t> packet(12, 0);
  packet[0] = 0x80;
  packet[1] = static_cast<uint8_t>(96 | (is_efu ? 0x80 : 0));
  packet[2] = static_cast<uint8_t>(*sequence >> 8);
  packet[3] = static_cast<uint8_t>((*sequence)++);
  packet[4] = static_cast<uint8_t>(timestamp >> 24);
  packet[5] = static_cast<uint8_t>(timestamp >> 16);
  packet[6] = static_cast<uint8_t>(timestamp >> 8);
  packet[7] = static_cast<uint8_t>(timestamp);
  constexpr uint32_t kSsrc = 0x12345678;
  packet[8] = static_cast<uint8_t>(kSsrc >> 24);
  packet[9] = static_cast<uint8_t>(kSsrc >> 16);
  packet[10] = static_cast<uint8_t>(kSsrc >> 8);
  packet[11] = static_cast<uint8_t>(kSsrc);
  packet.insert(packet.end(), payload.begin(), payload.end());

  sockaddr_in destination{};
  destination.sin_family = AF_INET;
  destination.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  destination.sin_port = htons(port);
  const ssize_t sent = ::sendto(socket, packet.data(), packet.size(), 0,
                                reinterpret_cast<const sockaddr*>(&destination),
                                sizeof(destination));
  assert(sent == static_cast<ssize_t>(packet.size()));
}

void ExpectFrame(mlvc::io::RtpMlvcReceiver* receiver, int expected_index,
                 MlvcFrameType expected_type) {
  int frame_index = -1;
  MlvcFrameType frame_type = MlvcFrameType::kPFrame;
  int q_index = -1;
  std::vector<uint8_t> payload;
  assert(receiver->ReceiveFrame(&frame_index, &frame_type, &q_index, &payload));
  assert(frame_index == expected_index);
  assert(frame_type == expected_type);
  assert(q_index == 10);
}

void TestEosWaitsForReorderedFrames() {
  mlvc::io::RtpMlvcReceiver receiver(0);
  const int socket = ::socket(AF_INET, SOCK_DGRAM, 0);
  assert(socket >= 0);
  uint16_t sequence = 1000;
  SendUnit(socket, receiver.local_port(), &sequence, 0, MakeScu(1));
  SendUnit(socket, receiver.local_port(), &sequence, 0,
           MakeEfu(1, 0, MlvcFrameType::kIFrame,
                   mlvc::transport::kMlvcNoReference,
                   mlvc::transport::kMlvcNoReference));
  (void)receiver.ReceiveHeader();
  ExpectFrame(&receiver, 0, MlvcFrameType::kIFrame);

  // Frame 2 and EOS complete before frame 1. The exclusive EOS boundary
  // requires the receiver to keep waiting instead of reporting normal EOF.
  SendUnit(socket, receiver.local_port(), &sequence, 6000,
           MakeEfu(1, 2, MlvcFrameType::kPFrame, 1,
                   mlvc::transport::kMlvcNoReference));
  SendUnit(socket, receiver.local_port(), &sequence, 9000,
           mlvc::transport::SerializeEos(1, 3));
  std::thread delayed_middle_frame([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    SendUnit(socket, receiver.local_port(), &sequence, 3000,
             MakeEfu(1, 1, MlvcFrameType::kPFrame, 0,
                     mlvc::transport::kMlvcNoReference));
  });

  ExpectFrame(&receiver, 1, MlvcFrameType::kPFrame);
  delayed_middle_frame.join();
  ExpectFrame(&receiver, 2, MlvcFrameType::kPFrame);
  int frame_index = -1;
  MlvcFrameType frame_type = MlvcFrameType::kPFrame;
  int q_index = -1;
  std::vector<uint8_t> payload;
  assert(!receiver.ReceiveFrame(&frame_index, &frame_type, &q_index, &payload));
  ::close(socket);
}

void TestCompletedFrameDuplicateIsIgnored() {
  mlvc::io::RtpMlvcReceiver receiver(0);
  const int socket = ::socket(AF_INET, SOCK_DGRAM, 0);
  assert(socket >= 0);
  uint16_t sequence = 2000;
  SendUnit(socket, receiver.local_port(), &sequence, 0, MakeScu(1));
  const auto first_frame = MakeEfu(1, 0, MlvcFrameType::kIFrame,
                                  mlvc::transport::kMlvcNoReference,
                                  mlvc::transport::kMlvcNoReference);
  SendUnit(socket, receiver.local_port(), &sequence, 0, first_frame);
  (void)receiver.ReceiveHeader();
  ExpectFrame(&receiver, 0, MlvcFrameType::kIFrame);

  // A completed logical unit retransmitted under a new RTP sequence must not
  // be handed to the decoder a second time or break the next frame.
  SendUnit(socket, receiver.local_port(), &sequence, 0, first_frame);
  SendUnit(socket, receiver.local_port(), &sequence, 3000,
           MakeEfu(1, 1, MlvcFrameType::kPFrame, 0,
                   mlvc::transport::kMlvcNoReference));
  SendUnit(socket, receiver.local_port(), &sequence, 6000,
           mlvc::transport::SerializeEos(1, 2));
  ExpectFrame(&receiver, 1, MlvcFrameType::kPFrame);
  int frame_index = -1;
  MlvcFrameType frame_type = MlvcFrameType::kPFrame;
  int q_index = -1;
  std::vector<uint8_t> payload;
  assert(!receiver.ReceiveFrame(&frame_index, &frame_type, &q_index, &payload));
  ::close(socket);
}

void TestDamagedEfuDoesNotTerminateReceiver() {
  mlvc::io::RtpMlvcReceiver receiver(0);
  const int socket = ::socket(AF_INET, SOCK_DGRAM, 0);
  assert(socket >= 0);
  uint16_t sequence = 3000;
  SendUnit(socket, receiver.local_port(), &sequence, 0, MakeScu(1));
  SendUnit(socket, receiver.local_port(), &sequence, 0,
           MakeEfu(1, 0, MlvcFrameType::kIFrame,
                   mlvc::transport::kMlvcNoReference,
                   mlvc::transport::kMlvcNoReference));
  (void)receiver.ReceiveHeader();
  ExpectFrame(&receiver, 0, MlvcFrameType::kIFrame);

  auto damaged = MakeEfu(1, 1, MlvcFrameType::kPFrame, 0,
                         mlvc::transport::kMlvcNoReference);
  damaged.back() ^= 0x80;
  SendUnit(socket, receiver.local_port(), &sequence, 3000, damaged);
  SendUnit(socket, receiver.local_port(), &sequence, 6000,
           MakeEfu(1, 2, MlvcFrameType::kIFrame,
                   mlvc::transport::kMlvcNoReference,
                   mlvc::transport::kMlvcNoReference));
  ExpectFrame(&receiver, 2, MlvcFrameType::kIFrame);

  SendUnit(socket, receiver.local_port(), &sequence, 9000,
           mlvc::transport::SerializeEos(1, 3));
  int frame_index = -1;
  MlvcFrameType frame_type = MlvcFrameType::kPFrame;
  int q_index = -1;
  std::vector<uint8_t> payload;
  assert(!receiver.ReceiveFrame(&frame_index, &frame_type, &q_index, &payload));
  ::close(socket);
}

void TestRandomAccessReorderKeepsLatePredecessor() {
  mlvc::io::RtpMlvcReceiver receiver(0);
  const int socket = ::socket(AF_INET, SOCK_DGRAM, 0);
  assert(socket >= 0);
  uint16_t sequence = 4000;
  SendUnit(socket, receiver.local_port(), &sequence, 0, MakeScu(1));
  SendUnit(socket, receiver.local_port(), &sequence, 0,
           MakeEfu(1, 0, MlvcFrameType::kIFrame,
                   mlvc::transport::kMlvcNoReference,
                   mlvc::transport::kMlvcNoReference));
  (void)receiver.ReceiveHeader();
  ExpectFrame(&receiver, 0, MlvcFrameType::kIFrame);

  SendUnit(socket, receiver.local_port(), &sequence, 6000,
           MakeEfu(1, 2, MlvcFrameType::kIFrame,
                   mlvc::transport::kMlvcNoReference,
                   mlvc::transport::kMlvcNoReference));
  std::thread delayed_middle_frame([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    SendUnit(socket, receiver.local_port(), &sequence, 3000,
             MakeEfu(1, 1, MlvcFrameType::kPFrame, 0,
                     mlvc::transport::kMlvcNoReference));
    SendUnit(socket, receiver.local_port(), &sequence, 9000,
             mlvc::transport::SerializeEos(1, 3));
  });

  ExpectFrame(&receiver, 1, MlvcFrameType::kPFrame);
  delayed_middle_frame.join();
  ExpectFrame(&receiver, 2, MlvcFrameType::kIFrame);
  int frame_index = -1;
  MlvcFrameType frame_type = MlvcFrameType::kPFrame;
  int q_index = -1;
  std::vector<uint8_t> payload;
  assert(!receiver.ReceiveFrame(&frame_index, &frame_type, &q_index, &payload));
  ::close(socket);
}

void TestLongGopGapWaitsForRecovery() {
  mlvc::io::RtpMlvcReceiver receiver(0);
  const int socket = ::socket(AF_INET, SOCK_DGRAM, 0);
  assert(socket >= 0);
  uint16_t sequence = 5000;
  SendUnit(socket, receiver.local_port(), &sequence, 0, MakeScu(1));
  SendUnit(socket, receiver.local_port(), &sequence, 0,
           MakeEfu(1, 0, MlvcFrameType::kIFrame,
                   mlvc::transport::kMlvcNoReference,
                   mlvc::transport::kMlvcNoReference));
  (void)receiver.ReceiveHeader();
  ExpectFrame(&receiver, 0, MlvcFrameType::kIFrame);

  for (uint32_t frame = 2; frame < 96; ++frame) {
    SendUnit(socket, receiver.local_port(), &sequence, frame * 3000,
             MakeEfu(1, frame, MlvcFrameType::kPFrame, frame - 1,
                     mlvc::transport::kMlvcNoReference));
  }
  SendUnit(socket, receiver.local_port(), &sequence, 96 * 3000,
           MakeEfu(1, 96, MlvcFrameType::kIFrame,
                   mlvc::transport::kMlvcNoReference,
                   mlvc::transport::kMlvcNoReference));
  SendUnit(socket, receiver.local_port(), &sequence, 97 * 3000,
           mlvc::transport::SerializeEos(1, 97));

  ExpectFrame(&receiver, 96, MlvcFrameType::kIFrame);
  int frame_index = -1;
  MlvcFrameType frame_type = MlvcFrameType::kPFrame;
  int q_index = -1;
  std::vector<uint8_t> payload;
  assert(!receiver.ReceiveFrame(&frame_index, &frame_type, &q_index, &payload));
  ::close(socket);
}

}  // namespace

int main() {
  mlvc::io::RtpMlvcReceiver receiver(0);
  const int socket = ::socket(AF_INET, SOCK_DGRAM, 0);
  assert(socket >= 0);
  uint16_t sequence = 100;

  SendUnit(socket, receiver.local_port(), &sequence, 0,
           MakeEfu(1, 0, MlvcFrameType::kIFrame, mlvc::transport::kMlvcNoReference,
                   mlvc::transport::kMlvcNoReference));
  // A receiver may see the RA EFU before its reordered/repeated SCU.
  SendUnit(socket, receiver.local_port(), &sequence, 0, MakeScu(1));
  const auto header = receiver.ReceiveHeader();
  assert(header.width == 1920 && header.height == 1080);
  ExpectFrame(&receiver, 0, MlvcFrameType::kIFrame);

  SendUnit(socket, receiver.local_port(), &sequence, 3000,
           MakeEfu(1, 1, MlvcFrameType::kPFrame, 0,
                   mlvc::transport::kMlvcNoReference, true));
  SendUnit(socket, receiver.local_port(), &sequence, 6000,
           MakeEfu(1, 2, MlvcFrameType::kPFrame, 1,
                   mlvc::transport::kMlvcNoReference));
  ExpectFrame(&receiver, 1, MlvcFrameType::kPFrame);
  ExpectFrame(&receiver, 2, MlvcFrameType::kPFrame);

  // Frame 3 is lost; frame 4 depends on it.  The explicit LTR recovery at 5
  // must bypass both the gap and the now-undecodable dependent frame 4.
  SendUnit(socket, receiver.local_port(), &sequence, 12000,
           MakeEfu(1, 4, MlvcFrameType::kPFrame, 3,
                   mlvc::transport::kMlvcNoReference));
  SendUnit(socket, receiver.local_port(), &sequence, 15000,
           MakeEfu(1, 5, MlvcFrameType::kLtrRecovery,
                   mlvc::transport::kMlvcNoReference, 1));
  ExpectFrame(&receiver, 5, MlvcFrameType::kLtrRecovery);

  SendUnit(socket, receiver.local_port(), &sequence, 18000,
           MakeEfu(1, 6, MlvcFrameType::kPFrame, 5,
                   mlvc::transport::kMlvcNoReference));
  ExpectFrame(&receiver, 6, MlvcFrameType::kPFrame);

  // Random access can recover even when the missing-frame distance is larger
  // than the ordinary reorder window.
  SendUnit(socket, receiver.local_port(), &sequence, 300000,
           MakeEfu(1, 100, MlvcFrameType::kIFrame,
                   mlvc::transport::kMlvcNoReference,
                   mlvc::transport::kMlvcNoReference));
  ExpectFrame(&receiver, 100, MlvcFrameType::kIFrame);

  // SCUs arrive in reverse order.  The receiver sorts them by config ID and
  // applies each at its own random-access frame boundary.
  SendUnit(socket, receiver.local_port(), &sequence, 303000, MakeScu(3));
  SendUnit(socket, receiver.local_port(), &sequence, 303000, MakeScu(2));
  SendUnit(socket, receiver.local_port(), &sequence, 303000,
           MakeEfu(2, 101, MlvcFrameType::kIFrame,
                   mlvc::transport::kMlvcNoReference,
                   mlvc::transport::kMlvcNoReference));
  SendUnit(socket, receiver.local_port(), &sequence, 306000,
           MakeEfu(3, 102, MlvcFrameType::kIFrame,
                   mlvc::transport::kMlvcNoReference,
                   mlvc::transport::kMlvcNoReference));
  ExpectFrame(&receiver, 101, MlvcFrameType::kIFrame);
  ExpectFrame(&receiver, 102, MlvcFrameType::kIFrame);

  // A delayed copy of an older SCU is harmless after a later config is active.
  SendUnit(socket, receiver.local_port(), &sequence, 306000, MakeScu(2));

  // The decoder only has frame 102 as its active short reference. The
  // receiver must reject an EFU that names a merely historical frame instead
  // of silently feeding the wrong feature tensor to the decoder.
  SendUnit(socket, receiver.local_port(), &sequence, 309000,
           MakeEfu(3, 103, MlvcFrameType::kPFrame, 101,
                   mlvc::transport::kMlvcNoReference));
  int bad_frame_index = -1;
  MlvcFrameType bad_frame_type = MlvcFrameType::kPFrame;
  int bad_q_index = -1;
  std::vector<uint8_t> bad_payload;
  bool rejected_unavailable_short_reference = false;
  try {
    (void)receiver.ReceiveFrame(&bad_frame_index, &bad_frame_type, &bad_q_index, &bad_payload);
  } catch (const std::exception&) {
    rejected_unavailable_short_reference = true;
  }
  assert(rejected_unavailable_short_reference);

  SendUnit(socket, receiver.local_port(), &sequence, 312000,
           MakeEfu(3, 103, MlvcFrameType::kPFrame, 102,
                   mlvc::transport::kMlvcNoReference));
  ExpectFrame(&receiver, 103, MlvcFrameType::kPFrame);

  SendUnit(socket, receiver.local_port(), &sequence, 315000,
           mlvc::transport::SerializeEos(3, 104));
  int frame_index = -1;
  MlvcFrameType frame_type = MlvcFrameType::kPFrame;
  int q_index = -1;
  std::vector<uint8_t> payload;
  assert(!receiver.ReceiveFrame(&frame_index, &frame_type, &q_index, &payload));
  ::close(socket);
  TestEosWaitsForReorderedFrames();
  TestCompletedFrameDuplicateIsIgnored();
  TestDamagedEfuDoesNotTerminateReceiver();
  TestRandomAccessReorderKeepsLatePredecessor();
  TestLongGopGapWaitsForRecovery();
  return 0;
}
