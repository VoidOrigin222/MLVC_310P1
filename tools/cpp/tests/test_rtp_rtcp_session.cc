#include "mlvc/transport/rtp_message_receiver.h"
#include "mlvc/transport/rtp_mlvc.h"

#include <cassert>
#include <arpa/inet.h>
#include <chrono>
#include <cstring>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#include <thread>
#include <vector>

using namespace mlvc::transport;

namespace {

int BindUdp(uint16_t port, sockaddr_in* bound_address) {
  const int socket = ::socket(AF_INET, SOCK_DGRAM, 0);
  assert(socket >= 0);
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = htons(port);
  assert(::bind(socket, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) == 0);
  socklen_t address_length = sizeof(*bound_address);
  assert(::getsockname(socket, reinterpret_cast<sockaddr*>(bound_address), &address_length) == 0);
  timeval timeout{};
  timeout.tv_sec = 2;
  assert(::setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0);
  return socket;
}

void TestNackRetransmitsCachedRtpPacket() {
  sockaddr_in rtp_address{};
  const int rtp_socket = BindUdp(0, &rtp_address);
  const uint16_t rtp_port = ntohs(rtp_address.sin_port);
  assert(rtp_port < 65535);
  sockaddr_in rtcp_address{};
  const int rtcp_socket = BindUdp(static_cast<uint16_t>(rtp_port + 1), &rtcp_address);
  constexpr uint32_t kMediaSsrc = 0x10203040u;
  RtpMlvcSender sender("127.0.0.1", rtp_port, kMediaSsrc, 0, 4096,
                       1024u * 1024u, 1000, 96, std::chrono::milliseconds(100));

  MlvcScu scu;
  scu.config_id = 1;
  scu.coded_width = scu.visible_width = 64;
  scu.coded_height = scu.visible_height = 64;
  scu.codec_bundle_sha256.fill(0x3a);
  const auto scu_bytes = SerializeScu(scu);
  sender.SetSessionConfig(1, scu_bytes);
  MlvcEfu efu;
  efu.config_id = 1;
  efu.frame_id = 0;
  efu.frame_type = 0;
  efu.unit_flags |= kEfuRandomAccess | kEfuResetReference;
  efu.entropy_payload = {0x41, 0x42, 0x43};
  sender.SendUnit(RtpUnitType::kEfu, 0, 1, 0, 9000, SerializeEfu(efu));
  sender.Flush();

  std::vector<std::vector<uint8_t>> original_packets;
  for (int i = 0; i < 2; ++i) {
    uint8_t data[2048]{};
    const ssize_t received = ::recv(rtp_socket, data, sizeof(data), 0);
    assert(received > 0);
    original_packets.emplace_back(data, data + received);
  }
  assert(DecodeRtpPacket(original_packets[0]).ssrc == kMediaSsrc);

  uint8_t control_data[2048]{};
  sockaddr_in sender_rtcp_address{};
  socklen_t sender_address_length = sizeof(sender_rtcp_address);
  const ssize_t control_size = ::recvfrom(
      rtcp_socket, control_data, sizeof(control_data), 0,
      reinterpret_cast<sockaddr*>(&sender_rtcp_address), &sender_address_length);
  assert(control_size > 0);
  const std::vector<uint8_t> sender_report(control_data, control_data + control_size);
  const auto parsed_sr = DecodeRtcpCompound(sender_report);
  assert(parsed_sr.sender_report.has_value());
  assert(parsed_sr.sender_report->sender_ssrc == kMediaSsrc);
  assert(parsed_sr.sender_report->packet_count == 2);

  const uint16_t lost_sequence = DecodeRtpPacket(original_packets[1]).sequence;
  RtcpReceiverReport rr;
  rr.sender_ssrc = 0x55667788u;
  rr.reports.push_back(RtcpReportBlock{kMediaSsrc, 0, 1, lost_sequence, 0, 0, 0});
  const auto nack = EncodeRtcpNack(RtcpNack{rr.sender_ssrc, kMediaSsrc, lost_sequence, 0});
  const auto feedback = EncodeRtcpReceiverReport(rr, "rtcp-nack-test", {nack});
  const ssize_t feedback_size = ::sendto(
      rtcp_socket, feedback.data(), feedback.size(), 0,
      reinterpret_cast<const sockaddr*>(&sender_rtcp_address), sender_address_length);
  assert(feedback_size == static_cast<ssize_t>(feedback.size()));

  uint8_t retransmitted[2048]{};
  const ssize_t retransmitted_size = ::recv(rtp_socket, retransmitted, sizeof(retransmitted), 0);
  assert(retransmitted_size > 0);
  const std::vector<uint8_t> retransmitted_packet(retransmitted,
                                                   retransmitted + retransmitted_size);
  assert(DecodeRtpPacket(retransmitted_packet).sequence == lost_sequence);
  assert(retransmitted_packet == original_packets[1]);
  sender.Flush();
  assert(sender.Stats().packets == 3);

  ::close(rtcp_socket);
  ::close(rtp_socket);
}

}  // namespace

int main() {
  constexpr uint32_t kMediaSsrc = 0x10203040u;
  RtpMessageReceiver receiver(0, "RTP/RTCP session test");
  RtpMlvcSender sender("127.0.0.1", receiver.local_port(), kMediaSsrc, 0, 4096,
                       1024u * 1024u, 1000, 96, std::chrono::milliseconds(100));

  MlvcScu scu;
  scu.config_id = 1;
  scu.coded_width = scu.visible_width = 64;
  scu.coded_height = scu.visible_height = 64;
  scu.codec_bundle_sha256.fill(0x3a);
  const auto scu_bytes = SerializeScu(scu);
  sender.SetSessionConfig(1, scu_bytes);

  MlvcEfu efu;
  efu.config_id = 1;
  efu.frame_id = 0;
  efu.frame_type = 0;
  efu.unit_flags |= kEfuRandomAccess | kEfuResetReference;
  efu.entropy_payload = {0x11, 0x22, 0x33};
  const auto efu_bytes = SerializeEfu(efu);
  sender.SendUnit(RtpUnitType::kEfu, 0, 1, 0, 9000, efu_bytes);
  sender.Flush();

  std::vector<uint8_t> received;
  assert(receiver.Receive(&received) && received == scu_bytes);
  assert(receiver.Receive(&received) && received == efu_bytes);

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (std::chrono::steady_clock::now() < deadline &&
         (sender.Stats().rtcp_packets_sent == 0 ||
          sender.Stats().rtcp_packets_received == 0 ||
          receiver.rtcp_receiver_reports_sent() == 0)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  assert(sender.Stats().rtcp_packets_sent > 0);
  assert(sender.Stats().rtcp_packets_received > 0);
  assert(receiver.rtcp_sender_reports_received() > 0);
  assert(receiver.rtcp_receiver_reports_sent() > 0);
  TestNackRetransmitsCachedRtpPacket();
  return 0;
}
