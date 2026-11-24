#include <cassert>
#include <arpa/inet.h>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <sys/socket.h>
#include <sys/time.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "mlvc/io/udp_frame_transport.h"
#include "mlvc/transport/rtp_message_receiver.h"
#include "mlvc/transport/rtp_mlvc.h"

int main() {
  constexpr uint8_t kPayloadType = 110;
  mlvc::transport::RtpMessageReceiver receiver(0, "sender test receiver", kPayloadType);
  mlvc::transport::RtpMlvcSender sender("127.0.0.1", receiver.local_port(), 0x01020304u,
                                        0, 4096, 4u * 1024u * 1024u, 1000, kPayloadType,
                                        std::chrono::milliseconds(100));
  std::vector<uint8_t> input(5000);
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = static_cast<uint8_t>(i * 31u);
  mlvc::transport::MlvcEfu efu;
  efu.config_id = 7;
  efu.frame_id = 9;
  efu.frame_type = 0;
  efu.unit_flags |= mlvc::transport::kEfuRandomAccess | mlvc::transport::kEfuResetReference;
  efu.entropy_payload = input;
  mlvc::transport::MlvcScu initial_scu;
  initial_scu.config_id = 7;
  initial_scu.coded_width = initial_scu.visible_width = 1920;
  initial_scu.coded_height = initial_scu.visible_height = 1080;
  for (std::size_t i = 0; i < initial_scu.codec_bundle_sha256.size(); ++i) {
    initial_scu.codec_bundle_sha256[i] = static_cast<uint8_t>(0x90u + i);
  }
  const auto initial_scu_unit = mlvc::transport::SerializeScu(initial_scu);
  sender.SetSessionConfig(7, initial_scu_unit);
  const auto input_unit = mlvc::transport::SerializeEfu(efu);
  sender.SendUnit(mlvc::transport::RtpUnitType::kEfu, 0, 7, 9, 9000, input_unit);
  sender.Flush();
  const auto stats = sender.Stats();
  assert(stats.packets == 6);
  assert(stats.media_unit_bytes == initial_scu_unit.size() + input_unit.size());
  assert(stats.wire_bytes == initial_scu_unit.size() + 64 + input_unit.size() + 5 * 64);
  std::vector<uint8_t> output;
  bool complete = false;
  std::vector<uint8_t> ignored_scu;
  assert(receiver.Receive(&ignored_scu));
  assert(mlvc::transport::ParseMediaUnitHeader(ignored_scu).unit_type ==
         mlvc::transport::MlvcMediaUnitType::kScu);
  for (int i = 0; i < 8 && !complete; ++i) complete = receiver.Receive(&output);
  assert(complete);
  assert(output == input_unit);
  assert(receiver.lost_packets() == 0);
  assert(receiver.duplicate_packets() == 0);
  const auto rtcp_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (std::chrono::steady_clock::now() < rtcp_deadline &&
         (sender.Stats().rtcp_packets_sent == 0 ||
          sender.Stats().rtcp_packets_received == 0 ||
          receiver.rtcp_receiver_reports_sent() == 0)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  assert(sender.Stats().rtcp_packets_sent > 0);
  assert(sender.Stats().rtcp_packets_received > 0);
  assert(receiver.rtcp_sender_reports_received() > 0);
  assert(receiver.rtcp_receiver_reports_sent() > 0);

  mlvc::transport::RtpMlvcSender random_sender("127.0.0.1", receiver.local_port(), 0,
                                               0, 4096, 4u * 1024u * 1024u, 1000,
                                               kPayloadType);
  mlvc::transport::MlvcScu session_scu;
  session_scu.config_id = 42;
  session_scu.coded_width = session_scu.visible_width = 1920;
  session_scu.coded_height = session_scu.visible_height = 1080;
  for (std::size_t i = 0; i < session_scu.codec_bundle_sha256.size(); ++i) {
    session_scu.codec_bundle_sha256[i] = static_cast<uint8_t>(0xa0u + i);
  }
  const auto session_unit = mlvc::transport::SerializeScu(session_scu);
  random_sender.SetSessionConfig(42, session_unit);
  random_sender.ResendSessionConfig(9001);
  random_sender.Flush();

  // Pacing belongs to the sender thread: enqueueing a unit must not sleep on
  // the encoder thread, while Flush waits for paced transmission to finish.
  mlvc::transport::RtpMlvcSender paced("127.0.0.1", receiver.local_port(), 0x01020305u,
                                       80000, 1240, 64 * 1024, 2000, kPayloadType);
  const auto pacing_begin = std::chrono::steady_clock::now();
  efu.config_id = 8;
  efu.frame_id = 10;
  efu.entropy_payload = input;
  initial_scu.config_id = 8;
  const auto paced_scu = mlvc::transport::SerializeScu(initial_scu);
  paced.SetSessionConfig(8, paced_scu);
  const auto paced_unit = mlvc::transport::SerializeEfu(efu);
  paced.SendUnit(mlvc::transport::RtpUnitType::kEfu, 0, 8, 10, 12000, paced_unit);
  const auto enqueue_elapsed = std::chrono::steady_clock::now() - pacing_begin;
  assert(enqueue_elapsed < std::chrono::milliseconds(100));
  paced.Flush();
  const auto pacing_elapsed = std::chrono::steady_clock::now() - pacing_begin;
  assert(pacing_elapsed >= std::chrono::milliseconds(350));
  assert(paced.Stats().packets == 6);

  const int timestamp_socket = ::socket(AF_INET, SOCK_DGRAM, 0);
  assert(timestamp_socket >= 0);
  timeval receive_timeout{};
  receive_timeout.tv_sec = 2;
  assert(::setsockopt(timestamp_socket, SOL_SOCKET, SO_RCVTIMEO, &receive_timeout,
                      sizeof(receive_timeout)) == 0);
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  assert(::bind(timestamp_socket, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) ==
         0);
  socklen_t address_size = sizeof(address);
  assert(::getsockname(timestamp_socket, reinterpret_cast<sockaddr*>(&address), &address_size) ==
         0);
  {
    mlvc::io::RtpMlvcSender timestamp_sender("127.0.0.1", ntohs(address.sin_port));
    mlvc::io::MlvcBitstreamHeader header;
    header.version = 3;
    header.width = 1920;
    header.height = 1080;
    header.fps = 25.0;
    header.q_index = 18;
    header.gop = 96;
    header.reset_interval = 32;
    header.ltr_start_idx = 8;
    header.ltr_period = 64;
    header.ltr_qp_shift = 8;
    for (std::size_t i = 0; i < header.codec_bundle_sha256.size(); ++i) {
      header.codec_bundle_sha256[i] = static_cast<uint8_t>(0xb0u + i);
    }
    timestamp_sender.SendHeader(header);
    timestamp_sender.SendFrame(0, mlvc::codec::MlvcFrameType::kIFrame, 18, {1, 2});
    mlvc::io::MlvcFrameMetadata p_metadata;
    p_metadata.explicit_metadata = true;
    p_metadata.model_q_index = 18;
    p_metadata.unit_flags = mlvc::transport::kEfuCrcPresent;
    p_metadata.short_ref_frame_id = 0;
    p_metadata.pts = 3600;
    timestamp_sender.SendFrame(1, mlvc::codec::MlvcFrameType::kPFrame, 18, p_metadata, {3, 4});
    bool frame_gap_rejected = false;
    try {
      timestamp_sender.SendFrame(3, mlvc::codec::MlvcFrameType::kPFrame, 18, {5, 6});
    } catch (const std::exception&) {
      frame_gap_rejected = true;
    }
    assert(frame_gap_rejected);
    timestamp_sender.Close();
  }
  std::vector<uint32_t> timestamps;
  for (int index = 0; index < 3; ++index) {
    uint8_t packet_bytes[2048]{};
    const ssize_t received = ::recv(timestamp_socket, packet_bytes, sizeof(packet_bytes), 0);
    assert(received > 0);
    const std::vector<uint8_t> packet(packet_bytes, packet_bytes + received);
    const auto parsed = mlvc::transport::DecodeRtpPacket(packet);
    timestamps.push_back(parsed.timestamp);
    const auto fragment = mlvc::transport::DecodeRtpMlvcFragment(parsed.payload);
    assert(parsed.marker == (fragment.unit_type == mlvc::transport::RtpUnitType::kEfu));
  }
  ::close(timestamp_socket);
  assert(timestamps.size() == 3);
  assert(timestamps[0] == timestamps[1]);
  assert(static_cast<uint32_t>(timestamps[2] - timestamps[1]) == 3600);
  return 0;
}
