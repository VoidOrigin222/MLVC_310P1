#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdint>
#include <algorithm>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mlvc/core/status.h"
#include "mlvc/transport/udp_message_transport.h"

namespace {

constexpr uint8_t kMagic0 = 0xeb;
constexpr uint8_t kMagic1 = 0x90;
constexpr uint8_t kTail0 = 0xcd;
constexpr uint8_t kTail1 = 0xde;
constexpr std::size_t kHeaderBytes = 14;
constexpr std::size_t kTailBytes = 2;
constexpr std::size_t kPayloadBytes = 1024;

void PutU16(std::vector<uint8_t>* packet, std::size_t offset, uint16_t value) {
  (*packet)[offset] = static_cast<uint8_t>(value);
  (*packet)[offset + 1] = static_cast<uint8_t>(value >> 8);
}

void PutU32(std::vector<uint8_t>* packet, std::size_t offset, uint32_t value) {
  for (int index = 0; index < 4; ++index) {
    (*packet)[offset + static_cast<std::size_t>(index)] =
        static_cast<uint8_t>(value >> (8 * index));
  }
}

std::vector<uint8_t> MakePacket(uint16_t packet_index, uint16_t total_packets,
                                uint32_t total_size, std::size_t payload_size, uint8_t value,
                                uint8_t reserved = 0) {
  mlvc::Check(payload_size <= kPayloadBytes, "test payload is too large");
  std::vector<uint8_t> packet(kHeaderBytes + payload_size + kTailBytes, 0);
  packet[0] = kMagic0;
  packet[1] = kMagic1;
  PutU16(&packet, 2, packet_index);
  PutU16(&packet, 4, total_packets);
  PutU16(&packet, 6, static_cast<uint16_t>(payload_size));
  PutU32(&packet, 8, total_size);
  packet[12] = 0;
  packet[13] = reserved;
  std::fill(packet.begin() + static_cast<std::ptrdiff_t>(kHeaderBytes),
            packet.begin() + static_cast<std::ptrdiff_t>(kHeaderBytes + payload_size), value);
  packet[kHeaderBytes + payload_size] = kTail0;
  packet[kHeaderBytes + payload_size + 1] = kTail1;
  return packet;
}

int MakeSocket() {
  const int socket = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (socket < 0) throw std::runtime_error("failed to create UDP validation sender socket");
  return socket;
}

void SendPacket(int socket, uint16_t port, const std::vector<uint8_t>& packet) {
  sockaddr_in destination{};
  destination.sin_family = AF_INET;
  destination.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  destination.sin_port = htons(port);
  const ssize_t sent =
      ::sendto(socket, packet.data(), packet.size(), 0,
               reinterpret_cast<const sockaddr*>(&destination), sizeof(destination));
  if (sent != static_cast<ssize_t>(packet.size())) {
    throw std::runtime_error("failed to send malformed UDP validation packet");
  }
}

void ExpectReceiverReject(uint16_t port, const std::vector<std::vector<uint8_t>>& packets,
                          const char* description) {
  mlvc::transport::UdpMessageReceiver receiver(port, "UDP fragment validation test");
  const int sender = MakeSocket();
  try {
    for (const auto& packet : packets) SendPacket(sender, port, packet);
    bool rejected = false;
    try {
      (void)receiver.Receive();
    } catch (const std::exception&) {
      rejected = true;
    }
    ::close(sender);
    mlvc::Check(rejected, std::string("malformed UDP message was accepted: ") + description);
  } catch (...) {
    ::close(sender);
    throw;
  }
}

}  // namespace

int main() {
  try {
    const uint16_t base_port = static_cast<uint16_t>(43000 + (getpid() % 1000));
    ExpectReceiverReject(base_port,
                         {MakePacket(0, 1, 2048, kPayloadBytes, 0x11)},
                         "fragment count does not match total size");
    ExpectReceiverReject(static_cast<uint16_t>(base_port + 1),
                         {MakePacket(0, 2, 2048, 1, 0x22)},
                         "non-final fragment is not full sized");
    ExpectReceiverReject(static_cast<uint16_t>(base_port + 2),
                         {MakePacket(0, 2, 2048, kPayloadBytes, 0x33),
                          MakePacket(1, 2, 2048, 1, 0x44)},
                         "final fragment length is incorrect");
    ExpectReceiverReject(static_cast<uint16_t>(base_port + 3),
                         {MakePacket(0, 2, 2048, kPayloadBytes, 0x55),
                          MakePacket(0, 2, 2048, kPayloadBytes, 0x66)},
                         "duplicate fragment contents differ");
    ExpectReceiverReject(static_cast<uint16_t>(base_port + 4),
                         {MakePacket(0, 1, 1024, kPayloadBytes, 0x77, 1)},
                         "reserved header byte is non-zero");

    std::cout << "UDP fragment validation test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_udp_fragment_validation failed: " << error.what() << "\n";
    return 1;
  }
}
